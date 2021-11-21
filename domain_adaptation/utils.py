from dataclasses import dataclass

import abc
import os
from enum import Enum
from typing import Dict, Iterable, Iterator, Optional, List, Set, Tuple, Union

import torch
from torch.utils.data import IterableDataset
import tqdm
from transformers import BatchEncoding, TrainingArguments


class Head(Enum):
    SEQ_CLASSIFICATION = 1
    TOKEN_CLASSIFICATION = 2
    LANGUAGE_MODEL = 3
    UNKNOWN = 4


class Schedule(Enum):
    COMBINED = 1
    SEQUENTIAL = 2
    DYNAMIC = 3


class StoppingStrategy(Enum):
    FIRST_OBJECTIVE_CONVERGED = 1
    ALL_OBJECTIVES_CONVERGED = 2
    FIRST_OBJECTIVE_NUM_EPOCHS = 3
    ALL_OBJECTIVES_NUM_EPOCHS = 4
    MANUAL = 5


class AdaptationDataset(IterableDataset, abc.ABC):
    """
    United dataset for both sequence and token training, and both supervised and unsupervised objectives.
    """

    def __init__(self, length: Optional[int] = None):
        self.length = length

    def __getitem__(self, index: int) -> BatchEncoding:
        raise ValueError("We shouldn't ever get here?")

    def __len__(self):
        return self.length

    @staticmethod
    def iter_text_file_per_line(path: str) -> Iterable[str]:
        with open(path) as f:
            for l in f:
                yield l.strip()


class TransformerAdaptationDataset(AdaptationDataset):

    def __init__(self, batch_encoding_params: Iterable[Dict[str, torch.LongTensor]], length: Optional[int] = None,
                 objective_id: Optional[int] = -1):
        """
        :param batch_encoding_params: Arguments to be passed to BatchEncoding (input_ids, attention_mask, labels)
        """
        super().__init__(length)
        self.objective_id = objective_id
        self.batch_encoding_params = batch_encoding_params

    def __iter__(self) -> Iterator[BatchEncoding]:
        """
        Iterates over collated items of the dataset. The items are already collated by the specific Objective,
        so that Schedules can perform item-level sampling.
        :return: iterator over the samples of the dataset.
        """
        return iter(BatchEncoding({**encoding, **{"oid": self.objective_id}}) for encoding in self.batch_encoding_params)


class AdaptationArguments(TrainingArguments):

    fixed_adaptation_args = {
            "per_device_train_batch_size": 1,  # batching is done by Objective, no two distinct batches
            "per_device_eval_batch_size": 1,  # should be present in a single infer batch
            "per_gpu_train_batch_size": None,  # aggregation over multiple objectives can be done using
            "per_gpu_eval_batch_size": None,  # `gradient_accumulation_steps` > 1
            "do_predict": False,  # we do not want to mangle with multi-objective reports here,
                                  # models are separately reloadable
            "disable_tqdm": True,  # scheduler takes care of top-level terminal monitoring
            "max_steps": -1,  # max steps are used to dynamically set a learning rate,
                              # setting it to a default (-1) will set it to num_epochs*sum(objectives.dataset_length)
    }

    def __init__(self,
                 stopping_strategy: StoppingStrategy,
                 stopping_patience: Optional[int] = 10,
                 sample_converged_objectives: bool = False,
                 separate_heads: bool = False,
                 **kwargs):

        # novel arguments, w.r.t. original TrainingArguments
        self.stopping_strategy = stopping_strategy
        self.stopping_patience = stopping_patience
        self.use_converged_objectives = sample_converged_objectives
        self.separate_heads = separate_heads  # TODO

        # adjustments of the defaults expected by Scheduler
        unexpected_adjusted_args = [arg for arg in kwargs.keys() if arg in self.fixed_adaptation_args.keys()]
        if unexpected_adjusted_args:
            raise ValueError("You should not set these TrainingArgs for Adaptation: %s" % unexpected_adjusted_args)

        # set default values to fixed args
        kwargs = {**kwargs, **self.fixed_adaptation_args}
        super().__init__(**kwargs)
