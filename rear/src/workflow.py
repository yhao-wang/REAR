from .dataset import RearDataCollator, get_dataset
from .load_model import load_model_and_tokenizer
from .trainer import RearTrainer
from .args import get_train_args, Seq2SeqTrainingArguments, RearArguments


def training_process(
    args: RearArguments,
    training_args: Seq2SeqTrainingArguments,
):  
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = get_dataset(args.data, tokenizer, args.is_warm_up)
    data_collator = RearDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    # Initialize our Trainer
    trainer = RearTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        head_scaler=args.head_scaler,
        train_dataset=dataset,
        is_warm_up=args.is_warm_up,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()
    
    
def run_train():
    training_args, args = get_train_args()
    training_args.gradient_checkpointing_kwargs=dict(use_reentrant=False)
    training_process(args, training_args)