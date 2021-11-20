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
    should_stop: bool = False

    def __init__(self, objectives: List[Objective],
                 args: AdaptationArguments):
        self.objectives: Dict[int, Objective] = {id(o): o for o in objectives}

        self.args = args

    @abc.abstractmethod
    def _sample_datasets(self, split: str, epoch: int) -> Iterable[Dict[str, Any]]:
        """
        :return: An iterable over the Objectives corresponding to a single epoch.
        """
        pass

    def objectives_log(self, split: str) -> Dict[str, float]:
        out_logs = {}
        n_last_steps = self.args.logging_steps if split == "train" else self.args.eval_steps
        for objective in self.objectives.values():
            out_logs = {**out_logs, **objective.per_objective_log(split, n_last_steps)}

        return out_logs

    def _objective_converged(self, oid: int) -> bool:
        passed_patience_evals = len(self.objectives[oid].loss_history["eval"]) >= self.args.stopping_patience
        did_not_improve = max(self.objectives[oid].loss_history["eval"][:-self.args.stopping_patience]) >= \
                          max(self.objectives[oid].loss_history["eval"][-self.args.stopping_patience:])

        return passed_patience_evals and did_not_improve

    def _objective_passed_epochs(self, oid: int) -> bool:
        return self.objectives[oid].epoch > self.args.num_train_epochs

    def _should_stop(self) -> bool:
        # a number of epochs per all objectives is an upper-bound of the training duration
        obj_passed_epochs = [oid for oid in self.objectives.keys() if self._objective_passed_epochs(oid)]
        if len(obj_passed_epochs) == len(self.objectives):
            return True

        # if the upper bound does not apply, check for the user-selected stopping strategy,
        if self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                           StoppingStrategy.ALL_OBJECTIVES_CONVERGED):
            converged_objectives = [obj for obj in self.objectives.values() if obj.has_converged()]
            if converged_objectives:
                logger.warning("Converged objectives" % converged_objectives)
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_CONVERGED:
                return len(converged_objectives) > 0
            else:
                return len(converged_objectives) == len(self.objectives)

        elif self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                             StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS):
            logger.warning("Objectives that passed max_epochs: %s" % [self.objectives[o] for o in obj_passed_epochs])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS:
                return len(obj_passed_epochs) > 0
            else:
                return len(obj_passed_epochs) == len(self.objectives)

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
        self.should_stop = self._should_stop()
        if self.should_stop:
            logger.warning("Scheduler reached the given termination condition: %s" % self.args.stopping_strategy.name)

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

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        Combines datasets, per-batch.
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
            datasets_iter = self._sample_datasets(split, num_passed_epochs)
            for batch_encoding in datasets_iter:
                # stop on next requested batch, if we're in the should_stop state
                if self.should_stop:
                    return

                self.objectives_loss_queue.append((split, batch_encoding["oid"]))
                yield batch_encoding
                num_steps += 1

            num_passed_epochs += 1

    def iterable_dataset(self, split: str) -> TransformerAdaptationDataset:
        # upper-bound of the number of training steps is used for learning rate scheduling
        length_combined = int(sum(o.dataset_length[split] for o in self.objectives.values())
                              * self.args.num_train_epochs)

        return TransformerAdaptationDataset(self._combine_datasets(split), length_combined)


class SequentialSchedule(TrainingSchedule):

    label = "sequential"

    def _sample_datasets(self, split: str, epoch: int) -> Iterable[Dict[str, Any]]:
        for i, (oid, objective) in enumerate(self.objectives.items()):
            for batch_encoding in objective.get_dataset(split, i, self.args.device, epoch):
                yield batch_encoding


class StridedSchedule(TrainingSchedule):

    label = "strided"

    def _sample_datasets(self, split: str, epoch: int) -> Iterable[Dict[str, Any]]:
        all_dataset_iters = (iter(obj.get_dataset(split, i, self.args.device, epoch))
                             for i, (oid, obj) in enumerate(self.objectives.items()))
        for batch_encoding in itertools.chain(*zip(*all_dataset_iters)):
            yield batch_encoding
