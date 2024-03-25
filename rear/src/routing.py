import argparse
import math
from tqdm import tqdm
import json
from .utils import get_logger, EM_compute, F1_compute
from .inf import get_vllm_model, get_scores, get_model, get_perplexity


logger = get_logger(__name__)
THRESHOLD = 13


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['reliability', 'consistency'])
    parser.add_argument('--source', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--outfile', type=str) 
    args = parser.parse_args()
    return args


def load_data(source):
    with open(source) as f:
        data = json.load(f)
    return data


def reliability(dic):
    docs = dic['ctxs']
    response = get_scores(dic['question'], docs)
    rely_answer = max(response, key=lambda x: x['score'])['pred']
    ans_dict = {'pred': rely_answer}
    gold_ans = dic.pop('reference', None)
    if gold_ans is not None:
        ans_dict.update(dict(EM=EM_compute(gold_ans, rely_answer)))
        ans_dict.update(dict(F1=F1_compute(gold_ans, rely_answer)))
    return {
        'question': dic['question'],
        'reference': gold_ans,
        'ctxs': docs,
        'response': response,
        'rely_answer': ans_dict,
    }


def consistency(dic):
    response = get_perplexity(dic['question'], dic['ctxs'], dic['response'])
    dic.update(dict(response=response))
    kc_idx = None
    kc_score = None
    for idx, res in enumerate(response):
        kc_tmp = res['score'] + 9 * res['con']
        if res['score'] > THRESHOLD and (kc_score is None or kc_score < kc_tmp):
            kc_idx = idx
            kc_score = kc_tmp
    if kc_idx is None:
        ans_dict = dic['rely_answer']
    else:
        con_answer = response[kc_idx]['pred']
        ans_dict = {'pred': con_answer}
        gold_ans = dic['reference']
        if gold_ans is not None:
            ans_dict.update(dict(EM=EM_compute(gold_ans, con_answer)))
            ans_dict.update(dict(F1=F1_compute(gold_ans, con_answer)))
    dic.update(dict(con_answer=ans_dict))
    return dic


def s2(args):
    get_model(args.model_path)
    data = load_data(args.source)
    res = []
    em = 0
    f1 = 0
    for dic in data:
        res.append(consistency(dic))
        if res[-1]['reference'] is not None:
            em += res[-1]['con_answer']['EM']
            f1 += res[-1]['con_answer']['F1']
    print(f"EM: {em / len(data)}")
    print(f"F1: {f1 / len(data)}")
    with open(args.outfile, "w", encoding='utf-8') as f:
        json.dump(res, f)


def s1(args):
    get_vllm_model(args.model_path)
    data = load_data(args.source)
    res = []
    em = 0
    f1 = 0
    for dic in data:
        res.append(reliability(dic))
        if res[-1]['reference'] is not None:
            em += res[-1]['rely_answer']['EM']
            f1 += res[-1]['rely_answer']['F1']
    print(f"EM: {em / len(data)}")
    print(f"F1: {f1 / len(data)}")
    with open(args.outfile, "w", encoding='utf-8') as f:
        json.dump(res, f)


def inf():
    args = get_args()
    if args.phase == 'reliability':
        s1(args)
    else:
        s2(args)
