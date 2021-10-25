import logging
from typing import List, Dict, Tuple, Union

import torch
from transformers import Trainer, BatchEncoding

from .lang_module import LangModule
from .schedules import TrainingSchedule
from .utils import AdaptationArguments

logger = logging.getLogger()


class Adapter(Trainer):
    """
    1. Gets LangModule as common torch.nn.Module, do all the parallel wrapping, optimizer handling etc. (init())
    2. Provides samples in the ordering given by the Scheduler (get_dataloader())
    3. Logs per-objective (get_test_dataloader() + evaluation_loop())
    4.

    """

    permitted_args = ["args", "tokenizer", "compute_metrics", "callbacks", "optimizers"]
    eval_metrics_prefix = "eval"

    def __init__(self, lang_module: LangModule, schedule: TrainingSchedule, args: AdaptationArguments, **kwargs):
        unexpected_args = [k for k in kwargs.keys() if k not in self.permitted_args]
        if unexpected_args:
            raise ValueError("Adapter(**kwargs) got these unexpected kwargs: %s" % unexpected_args)

        self.schedule = schedule
        # TODO: maybe resolve datasets' worker_init_fn for multi-GPU support

        orig_callbacks = [] if "callbacks" not in kwargs else kwargs["callbacks"]

        super().__init__(model=lang_module,
                         args=args,
                         train_dataset=self.schedule.iterable_dataset(split="train"),
                         eval_dataset=self.schedule.iterable_dataset(split="eval"),
                         data_collator=self.flattened_collator,
                         compute_metrics=None,  # would require a static prediction format, but it varies
                         callbacks=orig_callbacks + [schedule.should_stop_check_callback()],
                         **kwargs)

    @staticmethod
    def flattened_collator(features: List[BatchEncoding]) -> BatchEncoding:
        """
        Objectives take care of their own data collation, so this collator just flattens the outputs of batch_size=1.
        Trainer should keep the passed `per_device_*_batch_size` even on multiGPU training, so no data is omitted.
        :return: loss and a placeholder of unused outputs, for compatibility
        """
        assert len(features) == 1

        return features[0]

    def compute_loss(self, model: LangModule, inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, None]]:
        labels = inputs["labels"] if "labels" in inputs else inputs["label"]

        outputs = model(**inputs)
        if self.label_smoother is not None:
            raise NotImplementedError()  # objective-dependent label smoothing is custom
            # loss = self.label_smoother(outputs, labels)
        else:
            loss = self.schedule.compute_loss(outputs, labels)

        mock_outputs = torch.tensor([-1, -1])
        return (loss, mock_outputs) if return_outputs else loss

    def log(self, logs: [Dict[str, float]]) -> None:
        is_eval_log = any(self.eval_metrics_prefix in log_key for log_key in logs)
        extended_logs = self.schedule.objectives_log(split="eval" if is_eval_log else "train")
        return super().log({**logs, **extended_logs})

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        logger.warning("Evaluating...")
        out = super(Adapter, self).evaluate(*args, **kwargs)
        if "metric_key_prefix" in kwargs:
            self.eval_metrics_prefix = kwargs["metric_key_prefix"]

        # refresh exhausted evaluation iteration for possible next evaluation
        self.eval_dataset = self.schedule.iterable_dataset("eval")

        return out
