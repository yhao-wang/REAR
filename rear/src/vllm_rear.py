from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from vllm.model_executor.parallel_utils.communication_op import (
        tensor_model_parallel_all_gather)
    from vllm.model_executor.sampling_metadata import SamplingMetadata
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.core.scheduler import SchedulerOutputs
    from vllm.outputs import RequestOutput
    from vllm.engine.llm_engine import LLMEngine
    from vllm.sequence import (Sequence, PromptLogprobs, SampleLogprobs, SequenceGroup, SamplerOutput,
                            SequenceStatus)
    from vllm.sequence import SequenceGroupOutput, SequenceOutput
    from vllm.model_executor.weight_utils import (default_weight_loader,
                                                hf_model_weights_iterator)
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.input_metadata import InputMetadata
    from vllm.model_executor.models.llama import KVCache, LlamaForCausalLM
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader import _MODEL_REGISTRY
    from vllm.outputs import CompletionOutput
    from vllm.model_executor.layers.sampler import (
        Sampler,
        _prune_hidden_states,
        _get_logits,
        _apply_logits_processors,
        _get_penalties,
        _apply_penalties,
        _get_temperatures,
        _get_top_p_top_k_min_p,
        _SAMPLING_EPS,
        _apply_top_p_top_k,
        _sample,
        _get_logprobs,
        _apply_min_p,
    )
except ImportError:
    import pkg_resources
    required_version = "0.2.3"
    try:
        vllm_version = pkg_resources.get_distribution("vllm").version

        if vllm_version != required_version:
            raise Exception(f"You need to use vllm version {required_version}. Your current version is {vllm_version}.")

    except pkg_resources.DistributionNotFound:
        raise Exception("vllm is not installed. Please install vllm version 0.2.3 to proceed.")

THRESHOLD, REL_TOK_ID, IRR_TOK_ID = None, None, None


class RankSampler(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.real_vocab_size = 32004

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)
        
        rank_logits = logits[:, -1]
        rank_logits = rank_logits.tolist()
        logits = logits[:, :self.real_vocab_size]

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)
        # Apply presence and frequency penalties.
        presence_penalties, frequency_penalties, repetition_penalties = (
            _get_penalties(sampling_metadata))
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        assert len(repetition_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, sampling_metadata,
                                    presence_penalties, frequency_penalties,
                                    repetition_penalties)

        # Apply temperature scaling.
        temperatures = _get_temperatures(sampling_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                                dtype=logits.dtype,
                                device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks, min_ps = _get_top_p_top_k_min_p(
            sampling_metadata, self.real_vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.real_vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
        if do_min_p:
            logits = _apply_min_p(logits, min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_rank_sampler_output(sample_results, sampling_metadata,
                                        prompt_logprobs, sample_logprobs, rank_logits)


def _process_model_outputs(
        self, output: SamplerOutput,
        scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
    # Update the scheduled sequence groups with the model outputs.
    scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
    for seq_group, outputs in zip(scheduled_seq_groups, output):
        self._process_sequence_group_outputs(seq_group, outputs)

    # Free the finished sequence groups.
    self.scheduler.free_finished_seq_groups()

    # Create the outputs.
    request_outputs: List[RequestOutput] = []
    for seq_group, outputs in zip((scheduled_seq_groups +
                    scheduler_outputs.ignored_seq_groups), output):
        request_output = RequestOutput.from_seq_group(seq_group)
        request_outputs.append(request_output)

    if self.log_stats:
        # Log the system stats.
        self._log_system_stats(scheduler_outputs.prompt_run,
                            scheduler_outputs.num_batched_tokens)
    return request_outputs
        
class RankSequenceOutput:
    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        rank_logit: float,
        logprobs: Dict[int, float],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs
        self.rank_logit = rank_logit


def _build_rank_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
    rank_logits: List[float],
) -> SamplerOutput:
    sampler_output = []
    for (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs, rank_logit) in zip(sampling_metadata.seq_groups,
                                       sample_results, prompt_logprobs,
                                       sample_logprobs, rank_logits):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids,
                                                      next_token_ids,
                                                      group_sample_logprobs):
            seq_outputs.append(
                RankSequenceOutput(seq_ids[parent_id], next_token_id, rank_logit, logprobs))
        sampler_output.append(
            SequenceGroupOutput(seq_outputs, group_prompt_logprobs))
    return sampler_output


@classmethod
def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
    # Get the top-n sequences.
    n = seq_group.sampling_params.n
    seqs = seq_group.get_seqs()
    if seq_group.sampling_params.use_beam_search:
        sorting_key = lambda seq: seq.get_beam_search_score(
            seq_group.sampling_params.length_penalty)
    else:
        sorting_key = lambda seq: seq.get_cumulative_logprob()
    sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
    top_n_seqs = sorted_seqs[:n]

    # Create the outputs.
    outputs: List[CompletionOutput] = []
    for seq in top_n_seqs:
        logprobs = seq.output_logprobs
        if seq_group.sampling_params.logprobs is None:
            # NOTE: We need to take care of this case because the sequence
            # always has the logprobs of the sampled tokens even if the
            # logprobs are not requested.
            logprobs = None
        finshed_reason = SequenceStatus.get_finished_reason(seq.status)
        output = CompletionOutput(seqs.index(seq), seq.output_text,
                                    seq.get_output_token_ids(),
                                    seq.get_cumulative_logprob(), logprobs,
                                    finshed_reason)
        outputs.append(output)

    # Every sequence in the sequence group should have the same prompt.
    prompt = seq_group.prompt
    prompt_token_ids = seq_group.prompt_token_ids
    prompt_logprobs = seq_group.prompt_logprobs
    finished = seq_group.is_finished()
    request_output = cls(seq_group.request_id, prompt, prompt_token_ids,
                prompt_logprobs, outputs, finished)
    request_output.rank_logit = top_n_seqs[0].rank_logits[0]
    return request_output


def append_token_id(
    self,
    token_id: int,
    rank_logit: float,
    logprobs: Dict[int, float],
) -> None:
    assert token_id in logprobs
    if not hasattr(self, "rank_logits"):
        self.rank_logits = [rank_logit]
        if rank_logit > THRESHOLD:
            token_id = REL_TOK_ID
        else:
            token_id = IRR_TOK_ID
        logprobs = {token_id: rank_logit}
    self._append_tokens_to_blocks([token_id])
    self.output_logprobs.append(logprobs)
    self.data.append_token_id(token_id, logprobs[token_id])
        

def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
    # Process prompt logprobs
    prompt_logprobs = outputs.prompt_logprobs
    if prompt_logprobs is not None:
        seq_group.prompt_logprobs = prompt_logprobs

    # Process samples
    samples = outputs.samples
    parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    existing_finished_seqs = seq_group.get_finished_seqs()
    parent_child_dict = {
        parent_seq.seq_id: []
        for parent_seq in parent_seqs
    }
    for sample in samples:
        parent_child_dict[sample.parent_seq_id].append(sample)
    # List of (child, parent)
    child_seqs: List[Tuple[Sequence, Sequence]] = []

    # Process the child samples for each parent sequence
    for parent in parent_seqs:
        child_samples: List[SequenceOutput] = parent_child_dict[
            parent.seq_id]
        if len(child_samples) == 0:
            # This parent sequence has no children samples. Remove
            # the parent sequence from the sequence group since it will
            # not be used in the future iterations.
            parent.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(parent.seq_id)
            self.scheduler.free_seq(parent)
            continue
        # Fork the parent sequence if there are multiple child samples.
        for child_sample in child_samples[:-1]:
            new_child_seq_id = next(self.seq_counter)
            child = parent.fork(new_child_seq_id)
            child.append_token_id(child_sample.output_token,
                                    child_sample.rank_logit,
                                    child_sample.logprobs)
            child_seqs.append((child, parent))
        # Continue the parent sequence for the last child sample.
        # We reuse the parent sequence here to reduce redundant memory
        # copies, especially when using non-beam search sampling methods.
        last_child_sample = child_samples[-1]
        parent.append_token_id(last_child_sample.output_token,
                                last_child_sample.rank_logit,
                                last_child_sample.logprobs,)
        child_seqs.append((parent, parent))

    for seq, _ in child_seqs:
        self._decode_sequence(seq, seq_group.sampling_params)
        self._check_stop(seq, seq_group.sampling_params)

    # Non-beam search case
    if not seq_group.sampling_params.use_beam_search:
        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        # NOTE: we need to fork the new sequences before freeing the
        # old sequences.
        for seq, parent in child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)
        return

    # Beam search case
    # Select the child sequences to keep in the sequence group.
    selected_child_seqs = []
    unselected_child_seqs = []
    beam_width = seq_group.sampling_params.best_of
    length_penalty = seq_group.sampling_params.length_penalty

    # Select the newly finished sequences with the highest scores
    # to replace existing finished sequences.
    # Tuple of (seq, parent, is_new)
    existing_finished_seqs = [(seq, None, False)
                                for seq in existing_finished_seqs]
    new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                            if seq.is_finished()]
    all_finished_seqs = existing_finished_seqs + new_finished_seqs
    # Sort the finished sequences by their scores.
    all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
        length_penalty=length_penalty,
        eos_token_id=self.tokenizer.eos_token_id),
                            reverse=True)
    for seq, parent, is_new in all_finished_seqs[:beam_width]:
        if is_new:
            # A newly generated child sequence finishes and has a high
            # score, so we will add it into the sequence group.
            selected_child_seqs.append((seq, parent))
    for seq, parent, is_new in all_finished_seqs[beam_width:]:
        if is_new:
            # A newly generated child sequence finishes but has a low
            # score, so we will not add it into the sequence group.
            # Additionally, if this sequence is a continuation of a
            # parent sequence, we will need remove the parent sequence
            # from the sequence group.
            unselected_child_seqs.append((seq, parent))
        else:
            # An existing finished sequence has a low score, so we will
            # remove it from the sequence group.
            seq_group.remove(seq.seq_id)

    # select the top beam_width sequences from the running
    # sequences for the next iteration to continue the beam
    # search.
    running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                            if not seq.is_finished()]
    # Sort the running sequences by their scores.
    running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
        length_penalty=length_penalty,
        eos_token_id=self.tokenizer.eos_token_id),
                            reverse=True)

    # Check if we can stop the beam search.
    if len(running_child_seqs) == 0:
        # No running sequences, stop the beam search.
        stop_beam_search = True
    elif len(all_finished_seqs) < beam_width:
        # Not enough finished sequences, continue the beam search.
        stop_beam_search = False
    else:
        # Check the early stopping criteria
        best_running_seq = running_child_seqs[0][0]
        current_worst_seq = all_finished_seqs[beam_width - 1][0]
        stop_beam_search = self._check_beam_search_early_stopping(
            seq_group.sampling_params.early_stopping,
            seq_group.sampling_params, best_running_seq, current_worst_seq)

    if stop_beam_search:
        # Stop the beam search and remove all the running sequences from
        # the sequence group.
        unselected_child_seqs.extend(running_child_seqs)
    else:
        # Continue the beam search and select the top beam_width sequences
        # to continue the beam search.
        selected_child_seqs.extend(running_child_seqs[:beam_width])
        # The remaining running sequences will not be used in the next
        # iteration. Again, if these sequences are continuations of
        # parent sequences, we will need to remove the parent sequences
        # from the sequence group.
        unselected_child_seqs.extend(running_child_seqs[beam_width:])

    # For newly created child sequences, add them to the sequence group
    # and fork them in block manager if they are not finished.
    for seq, parent in selected_child_seqs:
        if seq is not parent:
            seq_group.add(seq)
            if not seq.is_finished():
                self.scheduler.fork_seq(parent, seq)

    # Free the finished and selected parent sequences' memory in block
    # manager. Keep them in the sequence group as candidate output.
    for seq, parent in selected_child_seqs:
        if seq is parent and seq.is_finished():
            self.scheduler.free_seq(seq)

    # Remove the unselected parent sequences from the sequence group and
    # free their memory in block manager.
    for seq, parent in unselected_child_seqs:
        if seq is parent:
            # Remove the parent sequence if it is not selected for next
            # iteration
            seq_group.remove(seq.seq_id)
            self.scheduler.free_seq(seq)


class LlamaForRear(LlamaForCausalLM):

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        x = hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision)
        elements = list(x)
        
        for idx, (name, loaded_weight) in enumerate(elements):
            if name == 'lm_head.weight':
                lm_head_idx = idx
            elif name == 'rel_score.weight':
                rel_score_idx = idx
        elements[lm_head_idx][1][-1:] = elements[rel_score_idx][1]
        elements.pop(rel_score_idx)
        for name, loaded_weight in elements:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


def enable_vllm_for_rearllama(threshold=13, rel_tok_id=32001, irr_tok_id=32002):
    global THRESHOLD, REL_TOK_ID, IRR_TOK_ID
    THRESHOLD = threshold
    REL_TOK_ID = rel_tok_id
    IRR_TOK_ID = irr_tok_id
    Sequence.append_token_id = append_token_id
    LLMEngine._process_sequence_group_outputs = _process_sequence_group_outputs
    LLMEngine._process_model_outputs = _process_model_outputs
    Sampler.__init__ = RankSampler.__init__
    Sampler.forward = RankSampler.forward
    RequestOutput.from_seq_group = from_seq_group           
    _MODEL_REGISTRY.update(dict(LlamaForRear=LlamaForRear))