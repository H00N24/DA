import inspect
import itertools
import logging
from typing import List, Dict, Any, Union

import torch
from transformers import Trainer, TrainingArguments, BatchEncoding

from domain_adaptation.lang_module import LangModule
from domain_adaptation.schedules import TrainingSchedule
from domain_adaptation.utils import AdaptationArguments

logger = logging.getLogger()


class Adapter(Trainer):
    """
    1. Gets LangModule as common torch.nn.Module, do all the parallel wrapping, optimizer handling etc. (init())
    2. Provides samples in the ordering given by the Scheduler (get_dataloader())
    3. Logs per-objective (get_test_dataloader() + evaluation_loop())
    4.

    """

    permitted_args = ["args", "tokenizer", "compute_metrics", "callbacks", "optimizers"]

    def __init__(self, lang_module: LangModule, schedule: TrainingSchedule, args: AdaptationArguments, **kwargs):
        unexpected_args = [k for k in kwargs.keys() if k not in self.permitted_args]
        if unexpected_args:
            raise ValueError("Adapter(**kwargs) got these unexpected kwargs: %s" % unexpected_args)

        self.schedule = schedule
        # TODO: per-objective logging
        # TODO: how to figure out a number of max_steps? Can that be somehow estimated?
        # TODO: maybe resolve datasets' worker_init_fn for multi-GPU support

        orig_callbacks = [] if "callbacks" not in kwargs else kwargs["callbacks"]

        super().__init__(model=lang_module,
                         args=args,
                         train_dataset=self.schedule.iterable_dataset(split="train"),
                         eval_dataset=self.schedule.iterable_dataset(split="eval"),
                         data_collator=self.flattened_collator,
                         callbacks=orig_callbacks + [schedule.should_stop_check_callback()],
                         **kwargs)

    @staticmethod
    def flattened_collator(features: List[BatchEncoding]) -> BatchEncoding:
        """
        Objectives take care of their own data collation, so this collator just flattens the outputs of batch_size=1.
        Trainer should keep the passed `per_device_*_batch_size` even on multiGPU training, so no data is omitted.
        :return:
        """
        assert len(features) == 1

        return features[0]

    def compute_loss(self, model: LangModule, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        labels = inputs["labels"] if "labels" in inputs else inputs["label"]

        outputs = model(**inputs)
        if self.label_smoother is not None:
            raise NotImplementedError()  # objective-dependent label smoothing is custom
            # loss = self.label_smoother(outputs, labels)
        else:
            loss = self.schedule.compute_loss(outputs, labels)

        return loss
