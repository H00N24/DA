import logging
import tqdm
from typing import List, Dict

from domain_adaptation.objectives.objective_base import Objective

logger = logging.getLogger()


class ScheduleState:

    main_bar: tqdm.trange
    bars: Dict[int, tqdm.trange] = {}
    epochs: Dict[int, int] = {}
    training_phase: str
    should_stop: bool = False

    train_loss_history: Dict[int, List[float]] = {}
    eval_loss_history: Dict[int, List[float]] = {}

    def __init__(self, objectives: List[Objective], schedule_label: str):
        self.train_loss_history = {id(o): [] for o in objectives}
        self.eval_loss_history = {id(o): [] for o in objectives}

        self.main_bar = tqdm.tqdm(desc="Training by %s schedule" % schedule_label,
                                  unit="updates",
                                  position=0,
                                  leave=True)

        for i, objective in enumerate(objectives):
            self.bars[id(objective)] = tqdm.trange(objective.dataset_length,
                                                   desc=objective.compatible_head.name,
                                                   unit="batches",
                                                   position=i+1,
                                                   leave=True)
            self.bars[id(objective)].set_postfix(epoch=0, loss=-1)
            self.epochs[id(objective)] = 0

    def update(self, oid: int, num_steps: int, loss: float):
        if self.bars[oid].last_print_n >= self.bars[oid].total:
            # reinitialize, with a new epoch
            self.epochs[oid] += 1
            self.bars[oid].set_postfix(epoch=self.epochs[oid], loss=loss)
            self.bars[oid].reset()

        self.main_bar.update(num_steps)
        self.bars[oid].update(num_steps)

        if self.training_phase == "train":
            self.train_loss_history[oid].append(loss)
        else:
            self.eval_loss_history[oid].append(loss)

    def change_training_phase(self, phase: str):
        assert phase in ("train", "eval")

        self.training_phase = phase
