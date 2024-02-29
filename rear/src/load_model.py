import torch
from transformers import AutoConfig, AutoTokenizer
from .modeling_rearllama import LlamaForRear


def load_model_and_tokenizer(
    args,
):
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    model_to_load = args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, trust_remote_code=True)

    special_token_dict = {"additional_special_tokens": ["<IRRELEVANT>", "<RELEVANT>", "<BEGIN_QUERY>", "<GENERATE_SCORE>"]}
    if tokenizer.pad_token_id is None:
        special_token_dict["pad_token"] = "<unk>"
    tokenizer.add_special_tokens(
        special_token_dict,
        replace_additional_special_tokens=False
    )
    config_kwargs = {
        "proj_scaler": args.proj_scaler,
        "bce_bias": args.bce_bias,
        "minor_diff": args.minor_diff,
        "is_warm_up": args.is_warm_up,
        "bias": args.rank_bias,
        "psg_num": args.psg_num,
        "beta": args.rank_beta,
        "gen_score_id": tokenizer.convert_tokens_to_ids("<GENERATE_SCORE>"),
        "rel_token_id": tokenizer.convert_tokens_to_ids("<RELEVANT>"),
        "irr_token_id": tokenizer.convert_tokens_to_ids("<IRRELEVANT>"),
        "trust_remote_code": True,
    }
        
    model = LlamaForRear.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model._keys_to_ignore_on_save = None
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = model.float()    
    model = model.train()

    return model, tokenizer
