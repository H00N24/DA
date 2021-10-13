import logging

import abc
import itertools
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import List, Iterable, Dict, Any

from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.schedule_state import ScheduleState
from domain_adaptation.utils import TransformerAdaptationDataset, StoppingStrategy, AdaptationArguments

logger = logging.getLogger()


class TrainingSchedule(abc.ABC):

    label: str

    objectives_loss_queue: List[int] = []

    def __init__(self, objectives: List[Objective],
                 args: AdaptationArguments,
                 training_phase: str = "train"):
        self.objectives: Dict[int, Objective] = {id(o): o for o in objectives}
        self.state = ScheduleState(objectives, self.label)
        self.state.change_training_phase(training_phase)

        self.args = args

    @abc.abstractmethod
    def _sample_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        :return:
        """
        pass

    def change_training_phase(self, phase: str) -> None:
        self.state.change_training_phase(phase)

    def _objective_converged(self, oid: int) -> bool:
        return len(self.state.eval_loss_history[oid]) >= self.args.stopping_patience \
                and max(self.state.eval_loss_history[oid][:-10]) >= max(self.state.eval_loss_history[oid][-10:])

    def _objective_passed_epochs(self, oid: int) -> bool:
        return self.state.epochs[oid] > self.args.num_train_epochs

    def _should_stop(self) -> bool:
        if self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_CONVERGES,
                                      StoppingStrategy.ALL_OBJECTIVES_CONVERGE):
            obj_converged = [oid for oid in self.objectives.keys() if self._objective_converged(oid)]
            logger.warning("Converged objectives" % [self.objectives[o] for o in obj_converged])
            if StoppingStrategy.FIRST_OBJECTIVE_CONVERGES:
                return len(obj_converged) > 0
            else:
                return len(obj_converged) == len(self.objectives)

        elif self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                        StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS):
            obj_passed_epochs = [oid for oid in self.objectives.keys() if self._objective_passed_epochs(oid)]
            logger.warning("Objectives that passed max_epochs: %s" % [self.objectives[o] for o in obj_passed_epochs])
            if StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS:
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
                """ Event called by Trainer after given `logging_steps`."""
                self.remember_if_should_stop()

        return AdaptationStoppingCallback()

    def remember_if_should_stop(self):
        self.state.should_stop = self.state.training_phase == "train" and self._should_stop()

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels) -> torch.FloatTensor:
        """
        TODO: assigns a loss of the corresponding objective, and possibly update sampling based on given schedule.
        :param labels:
        :param logit_outputs:
        :return:
        """
        oid = self.objectives_loss_queue.pop(0)
        # the objective loss arrives aggregated into a single item
        loss = self.objectives[oid].compute_loss(logit_outputs, labels)

        self.state.update(oid, 1, loss=loss.item())
        return loss

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        Combines datasets, per-batch.
        This main training iteration is upper-bound by a `num_epochs` over a full data set, defaults to infinite.
        :param split:
        :return:
        """
        # num_epoch over the whole dataset is an upper-bound of the iteration the combined data set,
        # but any chosen StoppingStrategy would terminate the training process earlier
        num_epochs_counter = self.args.num_train_epochs
        while num_epochs_counter is None or num_epochs_counter > 0:
            datasets_iter = self._sample_datasets(split)
            for batch_encoding in datasets_iter:
                # stop on next requested batch, if we're in should_stop state
                if self.state.should_stop:
                    return
                    # raise StopIteration()
                self.objectives_loss_queue.append(batch_encoding["oid"])
                yield batch_encoding
            num_epochs_counter -= 1

        # global num_epochs reached
        return
        # raise StopIteration()

    def iterable_dataset(self, split: str) -> TransformerAdaptationDataset:
        return TransformerAdaptationDataset(self._combine_datasets(split))


class SequentialSchedule(TrainingSchedule):

    label = "sequential"

    def _sample_datasets(self, split: str) -> Iterable[Iterable[Dict[str, Any]]]:
        for oid, objective in self.objectives.items():
            for batch_encoding in objective.get_dataset(split):
                yield batch_encoding


class StridedSchedule(TrainingSchedule):

    label = "strided"

    def _sample_datasets(self, split: str) -> Iterable[Iterable[Dict[str, Any]]]:
        all_dataset_iters = (iter(obj.get_dataset(split)) for oid, obj in self.objectives.items())
        for batch_encoding in itertools.chain(*zip(*all_dataset_iters)):
            yield batch_encoding
