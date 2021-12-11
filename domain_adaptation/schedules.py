import abc
import itertools
import logging
from typing import List, Iterable, Dict, Any, Tuple, Iterator, Optional, Sequence

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.utils import TransformerAdaptationDataset, StoppingStrategy, AdaptationArguments

logger = logging.getLogger()


class TrainingSchedule(abc.ABC):

    label: str
    objectives: Dict[str, Dict[int, Objective]]
    objectives_loss_queue: List[Tuple[str, int]]
    converged_objectives: List[Objective]
    should_stop: bool

    def __init__(self,
                 objectives: List[Objective],
                 args: AdaptationArguments,
                 extra_eval_objectives: Optional[List[Objective]] = ()):

        # eval objectives = train + eval => train objectives are evaluated implicitly
        self.objectives = {"train": {id(o): o for o in objectives},
                           "eval": {id(o): o for o in objectives + list(extra_eval_objectives)}}

        self.objectives_loss_queue = []
        self.converged_objectives = []
        self.should_stop = True

        self.args = args

    @abc.abstractmethod
    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        """
        :return: An iterable over the Objectives corresponding to a single epoch.
        """
        pass

    def objectives_log(self, split: str) -> Dict[str, float]:
        out_logs = {}
        for objective in self.objectives[split].values():
            out_logs = {**out_logs, **objective.per_objective_log(split)}

        return out_logs

    def _objective_passed_epochs(self, oid: int) -> bool:
        return self.objectives["train"][oid].epoch >= self.args.num_train_epochs

    def _should_stop(self) -> Tuple[bool, StoppingStrategy]:
        # a number of epochs per all objectives is an upper-bound of the training duration
        obj_passed_epochs = [oid for oid in self.objectives["train"].keys() if self._objective_passed_epochs(oid)]
        if len(obj_passed_epochs) == len(self.objectives["train"]):
            logger.warning("Scheduler reached the given maximum number of epochs for all objectives. "
                           "Triggering termination.")
            return True, StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS

        # if the upper bound does not apply, check for the user-selected stopping strategy,
        if self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                           StoppingStrategy.ALL_OBJECTIVES_CONVERGED):
            self.converged_objectives = [obj for obj in self.objectives["train"].values()
                                         if obj.has_converged(self.args.stopping_patience)]
            if self.converged_objectives:
                logger.warning("Converged objectives: %s" % [str(o) for o in self.converged_objectives])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_CONVERGED:
                return len(self.converged_objectives) > 0, self.args.stopping_strategy
            else:
                return len(self.converged_objectives) == len(self.objectives["train"]), self.args.stopping_strategy

        elif self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                             StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS):
            logger.warning("Objectives that passed max_epochs: %s" % [str(self.objectives["train"][o])
                                                                      for o in obj_passed_epochs])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS:
                return len(obj_passed_epochs) > 0, self.args.stopping_strategy
            else:
                return len(obj_passed_epochs) == len(self.objectives["train"]), self.args.stopping_strategy

        elif self.args.stopping_strategy == StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS:
            pass
        elif self.args.stopping_strategy != StoppingStrategy.MANUAL:
            pass

    def should_stop_check_callback(self) -> TrainerCallback:

        class AdaptationStoppingCallback(TrainerCallback):

            def on_log(cls, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
                """ Event called by Trainer after the given `logging_steps`."""
                self.remember_if_should_stop()

        return AdaptationStoppingCallback()

    def remember_if_should_stop(self):
        self.should_stop, stopping_strategy = self._should_stop()
        if self.should_stop:
            logger.warning("Scheduler reached a termination condition: %s" % stopping_strategy.name)

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels) -> torch.FloatTensor:
        """
        :param logit_outputs:
        :param labels:
        :return:
        """
        split, oid = self.objectives_loss_queue.pop(0)

        # the objective loss arrives aggregated into a single item
        loss = self.objectives[split][oid].compute_loss(logit_outputs, labels, split)

        return loss

    def _get_one_round_objective_sampler(self, objective: Objective, obj_i: int) -> Iterator[Dict[str, Any]]:
        """Default evaluation data sampling strategy"""
        dataset = objective.get_dataset("eval", obj_i, self.args.device)
        for sample in dataset:
            self.objectives_loss_queue.append(("eval", sample["oid"]))
            yield sample

    def _get_infinite_objective_sampler(self, objective: Objective, obj_i: int) -> Iterator[Dict[str, Any]]:
        """Default training data sampling strategy"""
        while True:
            # check for stopping conditions at the beginning of every objective epoch
            self.remember_if_should_stop()

            dataset = objective.get_dataset("train", obj_i, self.args.device)
            for sample in dataset:
                self.objectives_loss_queue.append(("train", sample["oid"]))
                yield sample

    def _sample_objective_dataset(self, objective: Objective, obj_i: int, split: str) -> Iterator[Dict[str, Any]]:
        if split == "train":
            # infinite iteration of the training resources, until the termination condition apply
            return self._get_infinite_objective_sampler(objective, obj_i)
        else:
            # single-round sampling - we do not want to iterate the evaluation forever
            return self._get_one_round_objective_sampler(objective, obj_i)

    def _combine_datasets_old(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        Combines datasets, based on the objectives sampling strategy, given by _sample_objective implementation.
        This main training iteration is upper-bound by a `num_epochs` over a full data set.
        :param split:
        :return:
        """
        # num_epoch over the whole dataset is an upper-bound of the iteration the combined data set,
        # but any chosen StoppingStrategy would terminate the training process earlier
        num_passed_epochs = 0
        num_steps = 0
        expected_num_epochs = self.args.num_train_epochs if split == "train" else 1
        while self.args.num_train_epochs is None or num_passed_epochs < expected_num_epochs:
            datasets_iter = self._sample_objectives(split)
            for batch_encoding in datasets_iter:
                # stop on next requested batch, if we're in the should_stop state
                if self.should_stop:
                    return

                self.objectives_loss_queue.append((split, batch_encoding["oid"]))
                yield batch_encoding
                num_steps += 1

            num_passed_epochs += 1

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        if split == "train":
            objective_sampler = self._sample_objectives(split)
        else:
            # evaluation split
            objective_sampler = SequentialSchedule.single_iteration_eval_sampling(self.objectives["eval"].values())

        objectives_data_samplers = {obj: self._sample_objective_dataset(obj, obj_i, split)
                                    for obj_i, obj in enumerate(self.objectives[split].values())}
        for i, objective in enumerate(objective_sampler):
            try:
                yield next(objectives_data_samplers[objective])
            except StopIteration:
                # TODO: evaluation routine sometimes raises StopIteration, we should find out why
                # logger.warning("Scheduler %s + Objective %s raised StopIteration.", self, objective)
                continue
            # stop on next requested batch, if we're in the should_stop state from on_log event
            if self.should_stop:
                return

    def iterable_dataset(self, split: str) -> TransformerAdaptationDataset:
        length_combined = int(sum((o.dataset_length[split] // o.batch_size)
                                  for o in self.objectives[split].values()))
        if split == "train":
            length_combined *= self.args.num_train_epochs

        return TransformerAdaptationDataset(self._combine_datasets(split), length_combined)


class SequentialSchedule(TrainingSchedule):

    label = "sequential"

    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        # infinite loop - does not determine a termination
        while True:
            for objective in self.objectives[split].values():
                for _ in range(objective.dataset_length[split]):
                    if objective in self.converged_objectives and not self.args.use_converged_objectives:
                        continue
                    yield objective

    @staticmethod
    def single_iteration_eval_sampling(objectives: Iterable[Objective]) -> Iterable[Objective]:
        for objective in objectives:
            for _ in range(objective.dataset_length["eval"]):
                yield objective


class StridedSchedule(TrainingSchedule):

    label = "strided"

    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        # infinite loop - does not determine a termination
        while True:
            for objective in self.objectives[split].values():
                if objective in self.converged_objectives and not self.args.use_converged_objectives:
                    continue
                yield objective
