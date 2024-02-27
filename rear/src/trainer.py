from transformers.trainer import *
from transformers import Seq2SeqTrainer


class RearTrainer(Seq2SeqTrainer):
# The substitue funtion for trainer._save_checkpoint
    def __init__(self, head_scaler: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_scaler = head_scaler
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            scaled_parameters = ["rel_score.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in scaled_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in scaled_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.head_scaler,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer