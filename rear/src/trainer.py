from transformers.trainer import *
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler


class RearTrainer(Seq2SeqTrainer):
# The substitue funtion for trainer._save_checkpoint
    def __init__(self, head_scaler: float = 1.0, *args, **kwargs):
        self.is_warm_up = kwargs.pop("is_warm_up", False)
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
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    
    def get_sampler(self):
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
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.is_warm_up:
            return super()._get_train_sampler()
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            # 对于分布式训练，我们保留DistributedSampler，但去掉seed参数，因为SequentialSampler不适用于分布式
            # 如果确实需要在分布式环境中保持数据顺序，这里的逻辑可能需要更复杂的定制
            return DistributedSampler(
                self.train_dataset,
                shuffle=False,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                # 移除seed参数，因为我们不打乱数据
            )
