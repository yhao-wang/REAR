import os
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import torch
from .utils import get_logger


logger = get_logger(__name__)


def process_ungrouped(sample, tokenizer):
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    source_ids = tokenizer.encode('<s>' + sample['input'], add_special_tokens=False)
    target_ids = tokenizer.encode(sample['output'] + '</s>', add_special_tokens=False)
    cutoff_len = 300
    max_target_len = min(15, len(target_ids))
    max_source_len = cutoff_len - max_target_len
    if len(source_ids) > max_source_len:
        split_id = tokenizer.convert_tokens_to_ids("<BEGIN_QUERY>")
        index = source_ids.index(split_id)
        document_ids, query_ids = source_ids[:index], source_ids[index:]
        query_max_length = min(64, len(query_ids))
        query_ids = query_ids[:query_max_length - 1] + [query_ids[-1]]
        document_max_length = max_source_len - len(query_ids)
        source_ids = document_ids[:document_max_length] + query_ids
            
    if len(target_ids) > max_target_len:
        target_ids = target_ids[:max_target_len]

    input_ids = source_ids + target_ids
    labels = [-100] *  len(source_ids) + target_ids
    
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "classes": [sample['score']],
    }
    
    
def process_grouped(samples, tokenizer):
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    feature = {"grouped_inputs": []}
    for sample in samples:
        source_ids = tokenizer.encode('<s>' + sample['input'], add_special_tokens=False)
        target_ids = tokenizer.encode(sample['output'] + '</s>', add_special_tokens=False)
        cutoff_len = 300
        max_target_len = min(15, len(target_ids))
        max_source_len = cutoff_len - max_target_len
        if len(source_ids) > max_source_len:
            split_id = tokenizer.convert_tokens_to_ids("<BEGIN_QUERY>")
            index = source_ids.index(split_id)
            document_ids, query_ids = source_ids[:index], source_ids[index:]
            query_max_length = min(64, len(query_ids))
            query_ids = query_ids[:query_max_length - 1] + [query_ids[-1]]
            document_max_length = max_source_len - len(query_ids)
            source_ids = document_ids[:document_max_length] + query_ids
                
        if len(target_ids) > max_target_len:
            target_ids = target_ids[:max_target_len]

        input_ids = source_ids + target_ids
        labels = [-100] *  len(source_ids) + target_ids
        
        feature.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "classes": [sample['score']],
        })
    return feature


def get_dataset(file_path, tokenizer, is_warm_up):
    if is_warm_up:
        process = process_ungrouped
    else:
        process = process_grouped
    dataset = load_dataset('json', data_files=file_path)
    train_dataset = dataset["train"]
    dataset_name = (file_path.split('/')[-1]).split('.')[0]
    os.makedirs(".cache", exist_ok=True)
    tokenized_dataset = train_dataset.map(process, fn_kwargs={'tokenizer': tokenizer}, num_proc=20, cache_file_name=f".cache/{dataset_name}-retriever.cache")
    return tokenized_dataset


class RearDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.judgment_toks = self.tokenizer.convert_tokens_to_ids(["<IRRELEVANT>", "<RELEVANT>"])

    def __call__(self, features, return_tensors=None):
        classes = []
        if "grouped_inputs" in features[0].keys():
            new_features = []
            for feature in features:
                new_features += feature["grouped_inputs"]
            features = new_features
        for feature in features:
            classes.append(feature.pop("classes"))
        features = super().__call__(features, return_tensors=return_tensors)
        for label_idx, label in enumerate(features['input_ids']):
            for idx, tok in enumerate(label):
                if tok in self.judgment_toks:
                    features['labels'][label_idx][idx] = self.label_pad_token_id
                    break
        features['classes'] = torch.tensor(classes, dtype=torch.float16)
        return features
