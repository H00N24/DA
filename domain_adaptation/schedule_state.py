import logging
import tqdm
from typing import List, Dict

from .objectives.objective_base import Objective

logger = logging.getLogger()


class ScheduleState:

    main_bar: tqdm.trange
    bars: Dict[int, tqdm.trange] = {}
    epochs: Dict[int, int] = {}
    training_phase: str
    should_stop: bool = False

    loss_history: Dict[str, Dict[int, List[float]]] = {"train": [], "eval": []}

    def __init__(self, objectives: List[Objective], schedule_label: str):
        self.loss_history["train"] = {id(o): [] for o in objectives}
        self.loss_history["eval"] = {id(o): [] for o in objectives}

        # self.main_bar = tqdm.tqdm(desc="Training by %s schedule" % schedule_label,
        #                           unit="updates",
        #                           position=0,
        #                           leave=True)

        for i, objective in enumerate(objectives):
            self.bars[id(objective)] = tqdm.trange(objective.dataset_length["train"],
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
            self.bars[oid].reset()
            self.bars[oid].set_postfix(epoch=self.epochs[oid], loss=loss)

        # self.main_bar.update(num_steps)
        self.bars[oid].update(num_steps)

        self.loss_history[self.training_phase][oid].append(loss)

    def change_training_phase(self, phase: str):
        assert phase in ("train", "eval")

        self.training_phase = phase

    def loss_summary(self, last_steps: int) -> Dict[str, float]:
        # TODO: propagate better objective label
        out_logs = {}
        for oid, history in self.loss_history[self.training_phase].items():
            out_logs["train %s" % oid] = sum(history[-last_steps:]) / last_steps

        return out_logs
