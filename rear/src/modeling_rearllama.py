""" PyTorch RankLLaMA model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
import transformers.models.llama.modeling_llama as LlamaModelFile
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaModel, LlamaFlashAttention2
from dataclasses import dataclass
from itertools import product

logger = logging.get_logger(__name__)


def flash_attn2_prepare_decoder_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_shape: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int
) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


LlamaModelFile.LlamaAttention = LlamaFlashAttention2
LlamaModel._prepare_decoder_attention_mask = flash_attn2_prepare_decoder_attention_mask


class LossForRank:

    def __init__(
        self,
        is_warm_up: bool = False,
        bias: float = 1.0,
        psg_num: int = 4,
        minor_diff: float = 0.0,
        bce_bias: float = 0.5,
    ):
        self.bias = bias
        self.compute_loss = F.binary_cross_entropy_with_logits 
        self.psg_num = psg_num
        self.minor_diff = minor_diff
        self.bce_bias = bce_bias
        if is_warm_up:
            self.loss_fn = self.coarse
        else:
            self.loss_fn = self.bi
        
    def __call__(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)
    
    def bi(self, **kwargs):
        return 0.5 * self.fine(**kwargs) + self.bce_bias * self.coarse(**kwargs)
        
    def fine(self, scores: torch.FloatTensor, labels: torch.FloatTensor):

        y_pred = scores.view(-1, self.psg_num)
        y_true = labels.view(-1, self.psg_num)
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]
        the_mask = (true_diffs > self.minor_diff) & (~torch.isinf(true_diffs))
        pred_diffs = pred_diffs[the_mask]

        true_diffs = true_diffs[the_mask]
        true_diffs = (true_diffs > 0).to(pred_diffs.dtype)

        return self.compute_loss(pred_diffs, true_diffs)
    
    def coarse(self, scores: torch.FloatTensor, labels: torch.FloatTensor):
        classes = labels[:] >= 0.5 
        pos_scores = scores[classes]
        neg_scores = scores[~classes]
        loss = torch.tensor(0, device=scores.device, dtype=scores.dtype)
        if pos_scores.shape[0]:
            loss += self.compute_loss(pos_scores, torch.ones_like(pos_scores), reduction="sum")
        if neg_scores.shape[0]:
            loss += self.bias * self.compute_loss(neg_scores, torch.zeros_like(neg_scores), reduction="sum")
        loss /= scores.shape[0]
        return loss

    
@dataclass
class RearOutput(ModelOutput):
    rel_scores: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: torch.Tensor = None
    hidden_states: torch.Tensor = None
    attentions: torch.Tensor = None


class LlamaForRear(LlamaForCausalLM):

    def __init__(
        self, 
        config: "LlamaConfig", 
        **kwargs,
        ):
        super().__init__(config)
        self._additional_init(hidden_size=config.hidden_size, **kwargs)

        self.post_init()      
    
    def _additional_init(
        self,
        hidden_size: int = 4096,
        gen_score_id: int = 32002,
        rel_token_id: int = 32001,
        irr_token_id: int = 32003,
        beta: float = 0.5, 
        is_warm_up : bool = False, 
        num_labels: int = 1, 
        enable_verify: bool = False, 
        bias: float = 0.7, 
        psg_num: int = 8, 
        minor_diff: float = 0.1,
        proj_scaler: float = 2.0,
        bce_bias: float = 0.3,
    ):
        self.num_labels = num_labels
        self.beta = beta
        self.rel_token_id = rel_token_id
        self.irr_token_id = irr_token_id
        self.rel_score = nn.Linear(hidden_size, num_labels, bias=False)
        self.gen_score_id = gen_score_id
        self.proj_scaler = proj_scaler
        self.rel_eval_fn = LossForRank(
            is_warm_up=is_warm_up,
            psg_num=psg_num,
            bias=bias,
            minor_diff=minor_diff,
            bce_bias=bce_bias,)
        self.enable_verify = enable_verify
        self.ans_fn = CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        classes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RearOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sft_output = self.ans(outputs=outputs, labels=labels)
        rank_output = self.rel_eval(input_ids=input_ids, outputs=outputs, classes=classes)
        
        loss = sft_output.loss + self.beta * rank_output.loss
        
        if not return_dict:
            output = (sft_output.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output 
        sft_output.rel_scores=rank_output.rel_scores
        return sft_output

    def ans(self, outputs, labels=None):
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss=torch.tensor(0, device=logits.device)
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = self.ans_fn(shift_logits, shift_labels)

        return RearOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def rel_eval(self, input_ids=None, outputs=None, classes=None):
        
        hidden_states = outputs[0]
        rel_position = (torch.eq(input_ids, self.gen_score_id).long().argmax(-1)).to(
            hidden_states.device
        )

        batch_size = hidden_states.size()[0]
        rel_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), rel_position]
        rel_logits = self.rel_score(rel_hidden_states)
        
        loss = None
        if classes is not None:
            loss = self.rel_eval_fn(scores=rel_logits, labels=classes.to(rel_logits.dtype))
        
        return RearOutput(
            loss=loss,
            rel_scores=rel_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        
   