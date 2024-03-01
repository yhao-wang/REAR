import argparse
from tqdm import tqdm
import json
from .utils import get_logger
from .inf import get_vllm_model, get_scores, get_model, get_perplexity


logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['reliability', 'consistency'])
    parser.add_argument('--source', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--outfile', type=str) 
    args = parser.parse_args()
    return args


def load_data(source):
    data = []
    with open(source) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def reliability(dic):
    docs = dic['dense_ctxs'][:10]
    response = get_scores(dic['question'], docs)
    return {
        'question': dic['question'],
        'reference': dic['reference'],
        'dense_ctxs': docs,
        'response': response,
    }


def consistency(dic):
    dic.update(dict(response=get_perplexity(dic['question'], dic['dense_ctxs'], dic['response'])))
    return dic


def s2(args):
    get_model(args.model_path)
    data = load_data(args.source)
    res = []
    for dic in tqdm(data):
        res.append(consistency(dic))
    with open(args.outfile, "w", encoding='utf-8') as f:
        json.dump(res, f)


def s1(args):
    get_vllm_model(args.model_path)
    data = load_data(args.source)
    res = []
    for dic in tqdm(data):
        res.append(reliability(dic))
    with open(args.outfile, "w", encoding='utf-8') as f:
        json.dump(res, f)


def inf():
    args = get_args()
    if args.phase == 'reliability':
        s1(args)
    else:
        s2(args)
