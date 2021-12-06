import logging
import os
from typing import List, Dict, Tuple, Union, Optional

from transformers import WEIGHTS_NAME
import torch
from transformers import Trainer, BatchEncoding
from transformers.modeling_utils import unwrap_model

from .lang_module import LangModule
from .schedules import TrainingSchedule
from .utils import AdaptationArguments

logger = logging.getLogger()


class Adapter(Trainer):
    """
    1. Gets LangModule as common torch.nn.Module, do all the parallel wrapping, optimizer handling etc. (init())
    2. Provides samples in the ordering given by the Scheduler (get_dataloader())
    3. Logs per-objective (get_test_dataloader() + evaluation_loop())
    4. Keeps the reference to the adapted model after the training process

    """

    permitted_args = ["args", "tokenizer", "callbacks", "optimizers"]
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
                         compute_metrics=None,  # would require a static prediction format among objectives
                         callbacks=orig_callbacks + [schedule.should_stop_check_callback()],
                         **kwargs)

    @staticmethod
    def flattened_collator(features: List[BatchEncoding]) -> BatchEncoding:
        """
        Objectives take care of their own data collation, so this collator just flattens the outputs of batch_size=1.
        Trainer should keep the passed `per_device_*_batch_size` even on multiGPU training, so no data is omitted.
        :return: loss and a placeholder of unused outputs, for compatibility
        """
        assert len(features) == 1  # TODO: on multi-gpu training, this does not hold

        return features[0]

    def compute_loss(self, model: LangModule, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[
        torch.FloatTensor, Tuple[torch.FloatTensor, None]]:
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

    def save_model(self, output_dir: Optional[str] = None) -> None:
        # HF native reload compatibility
        # TODO: low priority - we should persist some meta info with the objectives' modules,
        #  so if the same objective is used multiply, we can distinguish their persistence directories
        # for now we just increment suffix over the same objective types

        objectives_counter = {type(obj): 0 for obj in self.schedule.objectives["train"].values()}

        for objective_id, module in self.model.trainable_models.items():
            objective = self.schedule.objectives["train"][int(objective_id)]
            output_module_path = os.path.join(output_dir, str(objective))

            # if the objective of this type was already persisted, we'll index the configs of the next ones
            if objectives_counter[type(objective)] != 0:
                output_module_path += ("_" + objectives_counter[type(objective)])
                objectives_counter[type(objective)] += 1

            # we persist a shared tokenizer and training args either way
            self.model.tokenizer.save_pretrained(output_module_path)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

            if hasattr(module, "save_pretrained") or hasattr(unwrap_model(module), "save_pretrained"):
                # if the head module has "save_pretrained" method, it will be called for persistence
                module.save_pretrained(output_module_path, use_diff=True)
            else:
                # otherwise, we persist only a raw pytorch module
                torch.save(module.state_dict(), os.path.join(output_module_path, WEIGHTS_NAME))

            logger.info(f"Model of objective {str(objective)} saved in {output_module_path}")
