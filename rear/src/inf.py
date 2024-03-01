import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from .vllm_rear import enable_vllm_for_rankllama
from .utils import get_logger
from .modeling_rearllama import LlamaForCausalLM

logger = get_logger(__name__)


model, tokenizer, generation_config = None, None, None


def get_vllm_model(model_path, **kwargs):
    tmp_tokenizer = AutoTokenizer.from_pretrained(model_path)
    rel_tok_id = tmp_tokenizer.convert_tokens_to_ids("<RELEVANT>")
    irr_tok_id = tmp_tokenizer.convert_tokens_to_ids("<IRRELEVANT>")
    enable_vllm_for_rankllama(rel_tok_id=rel_tok_id, irr_tok_id=irr_tok_id)
    global model, tokenizer, generation_config
    gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.9)
    kwargs.update(dict(greedy=True, max_new_tokens=20))
    model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16", swap_space=8)
    tokenizer = model.get_tokenizer()
    generation_config = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=20)
    
def get_model(model_path, **kwargs):
    global model, tokenizer
    kwargs = {"torch_dtype": torch.bfloat16, 'trust_remote_code': True}
    model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    kwargs.update(dict(greedy=True, max_new_tokens=20))
    model = model.eval()

def batch_generate(prompt):
    outputs = model.generate(prompt, sampling_params=generation_config, use_tqdm=False)
    responses = [output.outputs[0].text.strip() for output in outputs]
    scores = [output.rank_logit for output in outputs]
    return responses, scores

def get_scores(query, ctxs):
    template = "{document}<BEGIN_QUERY>{query}<GENERATE_SCORE>"

    prompt = [template.format(document=doc, query=query) for doc in ctxs]
    responses, scores = batch_generate(prompt)

    res = []
    for pred, s in zip(responses, scores):
        res.append({"preds": pred,
                    "scores": s,})
    return res

def batch_perplexity(pairs):
    input_ids = []
    labels = []
    beta = model.beta
    model.beta = None
    for p in pairs:
        if p[1] == "":
            p[1] == "</s>"
        tokenized = [tokenizer(s, add_special_tokens=False).input_ids for s in p]
        i = tokenized[0] + tokenized[1][:-1]
        o = tokenized[1]
        input_ids.append({
            "input_ids": i,
        })
        if not o:
            o = [1]
        labels.append(o)
    inputs = tokenizer.pad(input_ids, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    loss = []
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        for i, label in enumerate(labels):
            res = logits[i, -len(label):, :].contiguous()
            label = torch.as_tensor(label, device="cuda")
            loss.append(F.cross_entropy(res, label).item())
    model.beta = beta 
    return loss

def get_perplexity(query, ctxs, res):
    batch_size = 10
    preds = [dic['preds'][0] for dic in res]
    q_template = "<s>{d}<BEGIN_QUERY>{q}<GENERATE_SCORE><IRRELEVANT>"
    a_template = "{a}"
    
    pairs = [(q_template.format(q=query, d=c), a_template.format(a=a)) for c, a in zip(ctxs, preds)]
    ppl_res = []
    for batch_id in range(0, len(pairs), batch_size):
        ppl_res += batch_perplexity(pairs)
    
    for did, dic in enumerate(res):
        dic["verification"] = []
        for res_idx in range(2):
            dic["verification"].append(ppl_res[did])
    return res