import logging

import abc
import itertools
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import List, Iterable, Dict, Any, Tuple, Union

from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.utils import TransformerAdaptationDataset, StoppingStrategy, AdaptationArguments

logger = logging.getLogger()


class TrainingSchedule(abc.ABC):

    label: str
    objectives_loss_queue: List[Tuple[str, int]] = []
    converged_objectives: List[Objective] = []
    should_stop: bool = False

    def __init__(self, objectives: List[Objective],
                 args: AdaptationArguments):
        self.objectives: Dict[int, Objective] = {id(o): o for o in objectives}

        self.args = args

    @abc.abstractmethod
    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        """
        :return: An iterable over the Objectives corresponding to a single epoch.
        """
        pass

    def objectives_log(self, split: str) -> Dict[str, float]:
        out_logs = {}
        for objective in self.objectives.values():
            n_last_steps = self.args.logging_steps if split == "train" \
                               else 1 \
                               # else (objective.dataset_length["eval"] // self.args.per_device_eval_batch_size)
            out_logs = {**out_logs, **objective.per_objective_log(split, aggregation_steps=n_last_steps)}

        return out_logs

    def _objective_converged(self, oid: int) -> bool:
        passed_patience_evals = len(self.objectives[oid].loss_history["eval"]) >= self.args.stopping_patience
        did_not_improve = max(self.objectives[oid].loss_history["eval"][:-self.args.stopping_patience]) >= \
                          max(self.objectives[oid].loss_history["eval"][-self.args.stopping_patience:])

        return passed_patience_evals and did_not_improve

    def _objective_passed_epochs(self, oid: int) -> bool:
        return self.objectives[oid].epoch >= self.args.num_train_epochs

    def _should_stop(self) -> Tuple[bool, StoppingStrategy]:
        # a number of epochs per all objectives is an upper-bound of the training duration
        obj_passed_epochs = [oid for oid in self.objectives.keys() if self._objective_passed_epochs(oid)]
        if len(obj_passed_epochs) == len(self.objectives):
            logger.warning("Scheduler reached the given maximum number of epochs for all objectives. "
                           "Triggering termination.")
            return True, StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS

        # if the upper bound does not apply, check for the user-selected stopping strategy,
        if self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                           StoppingStrategy.ALL_OBJECTIVES_CONVERGED):
            self.converged_objectives = [obj for obj in self.objectives.values()
                                         if obj.has_converged(self.args.stopping_patience)]
            if self.converged_objectives:
                logger.warning("Converged objectives: %s" % [str(o) for o in self.converged_objectives])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_CONVERGED:
                return len(self.converged_objectives) > 0, self.args.stopping_strategy
            else:
                return len(self.converged_objectives) == len(self.objectives), self.args.stopping_strategy

        elif self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                             StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS):
            logger.warning("Objectives that passed max_epochs: %s" % [str(self.objectives[o])
                                                                      for o in obj_passed_epochs])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS:
                return len(obj_passed_epochs) > 0, self.args.stopping_strategy
            else:
                return len(obj_passed_epochs) == len(self.objectives), self.args.stopping_strategy

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
        loss = self.objectives[oid].compute_loss(logit_outputs, labels, split)

        return loss

    def _get_one_round_objective_sampler(self, objective: Objective, obj_i: int, split: str) -> Iterable[Dict[str, Any]]:
        dataset = objective.get_dataset(split, obj_i, self.args.device)
        for sample in dataset:
            self.objectives_loss_queue.append((split, sample["oid"]))
            yield sample

    def _get_infinite_objective_sampler(self, objective: Objective, obj_i: int, split: str) -> Iterable[Dict[str, Any]]:
        while True:
            # check for stopping conditions at the beginning of every objective epoch
            self.remember_if_should_stop()

            dataset = objective.get_dataset(split, obj_i, self.args.device)
            for sample in dataset:
                self.objectives_loss_queue.append((split, sample["oid"]))
                yield sample

    def _sample_objective_dataset(self, objective: Objective, obj_i: int, split: str) -> Iterable[Dict[str, Any]]:
        if split == "train":
            # infinite iteration of the training resources, until the termination condition apply
            return self._get_infinite_objective_sampler(objective, obj_i, split)
        else:
            # single-round sampling - we do not want to iterate the evaluation forever
            return self._get_one_round_objective_sampler(objective, obj_i, split)

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
        obj_data_samplers = {obj: self._sample_objective_dataset(obj, obj_i, split)
                             for obj_i, obj in enumerate(self.objectives.values())}
        while not self.should_stop:
            for objective in self._sample_objectives(split):
                yield next(obj_data_samplers[objective])
                # stop on next requested batch, if we're in the should_stop state from on_log event
                if self.should_stop:
                    return
            # only the training iteration can be infinite
            if not split == "train":
                break

    def iterable_dataset(self, split: str) -> TransformerAdaptationDataset:
        # upper-bound of the number of training steps is used for learning rate scheduling
        dataset_epochs = self.args.num_train_epochs if split == "train" else 1
        length_combined = int(max((o.dataset_length[split] // o.batch_size) * dataset_epochs
                                  for o in self.objectives.values())) * len(self.objectives)

        return TransformerAdaptationDataset(self._combine_datasets(split), length_combined)


class SequentialSchedule(TrainingSchedule):

    label = "sequential"

    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        if split == "train":
            # infinite loop
            while True:
                for objective in self.objectives.values():
                    for _ in range(objective.dataset_length[split]):
                        if objective in self.converged_objectives and not self.args.use_converged_objectives:
                            continue
                        yield objective
        else:
            # single loop
            for objective in self.objectives.values():
                for _ in range(objective.dataset_length[split]):
                    yield objective


class StridedSchedule(TrainingSchedule):

    label = "strided"

    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        if split == "train":
            # infinite loop
            while True:
                for objective in self.objectives.values():
                    if objective in self.converged_objectives and not self.args.use_converged_objectives:
                        continue
                    yield objective
        else:
            # single loop
            for objective in self.objectives.values():
                yield objective
