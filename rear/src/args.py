from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from typing import Optional
from dataclasses import dataclass, field
import transformers


@dataclass
class RearArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    model_name_or_path: str = field(
        default=None,
    )
    data: str = field(
        default=None,
    )
    rank_beta: Optional[float] = field(
        default=None,
    )
    rank_bias: Optional[float] = field(
        default=1,
    )
    psg_num: Optional[int] = field(
        default=8,
    )
    is_warm_up: Optional[bool] = field(
        default=False,
    )
    minor_diff: Optional[float] = field(
        default=0.0,
    )
    head_scaler: Optional[float] = field(
        default=1.0,
    )
    proj_scaler: Optional[float] = field(
        default=1.0,
    )
    bce_bias: Optional[float] = field(
        default=0.5,
    )


def get_train_args():
    parser = HfArgumentParser((
        Seq2SeqTrainingArguments,
        RearArguments,
    ))
    training_args, rear_args = parser.parse_args_into_dataclasses()

    return training_args, rear_args 
