import abc
import itertools
import logging
from typing import List, Union, Optional, Iterable, Tuple, Dict, Sequence, Any

import torch
from tqdm import trange

from ..evaluators.evaluator_base import EvaluatorBase
from ..lang_module import LangModule
from ..utils import AdaptationDataset, Head, TransformerAdaptationDataset


logger = logging.getLogger()


class Objective(abc.ABC):

    compatible_head: Head
    given_id: str
    epoch: int

    texts: Optional[List[str]]
    texts_path: Optional[str]

    val_texts_path: Optional[str]
    val_texts: Optional[List[str]]

    dataset_length: Dict[str, int]
    loss_history: Dict[str, List[float]]
    outputs_history: Dict[str, List[Tuple[torch.FloatTensor, torch.LongTensor]]]
    evaluations_history: Dict[str, Dict[Union[str, EvaluatorBase], List[float]]]
    progressbar: Dict[str, trange]
    evaluators: Dict[str, List[EvaluatorBase]]

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 train_evaluators: Sequence[EvaluatorBase] = (),
                 val_evaluators: Sequence[EvaluatorBase] = (),
                 share_other_objective_head: Optional["Objective"] = None,
                 objective_module: Optional[torch.nn.Module] = None,
                 objective_id: Optional[str] = ""):
        self.batch_size = batch_size
        self.tokenizer = lang_module.tokenizer
        self.given_id = objective_id

        self.compatible_head_model = self.register_compatible_head_model(lang_module,
                                                                         share_other_objective_head,
                                                                         {},
                                                                         objective_module)
        self.epoch = 0
        self.dataset_length = {"train": 0, "eval": 0}
        self.loss_history = {"train": [], "eval": []}  # loss is treated differently than other outputs
        self.outputs_history = {"train": [], "eval": []}
        self.evaluators = {"train": [], "eval": []}
        self.evaluations_history = {"train": {}, "eval": {}}
        self.progressbar = {}

        self.texts = None
        self.val_texts = None
        self.texts_path = None
        self.val_texts_path = None

        if type(texts_or_path) == str:
            self.texts_path = texts_or_path
            with open(self.texts_path) as f:
                self.dataset_length["train"] = len(f.readlines())
        else:
            self.texts = texts_or_path
            self.dataset_length["train"] = len(self.texts)
        assert self.dataset_length, \
            "Objective %s was initialized with texts_or_path of zero length, this wouldn't work :("
        for split, given_evaluators in zip(("train", "eval"), (train_evaluators, val_evaluators)):
            for given_evaluator in given_evaluators:
                if given_evaluator.compatible_head != self.compatible_head:
                    raise ValueError("%s got incompatible evaluator: %s" % (self, given_evaluator))
                self.evaluators[split].append(given_evaluator)
                self.evaluations_history[split][given_evaluator] = []

            # loss is objective-dependent, hence we do not delegate it to a separate Evaluator object
            self.evaluations_history[split]["loss"] = []

        if val_texts_or_path is not None:
            if type(val_texts_or_path) == str:
                self.val_texts_path = val_texts_or_path
                with open(self.val_texts_path) as f:
                    self.dataset_length["eval"] = len(f.readlines())
            else:
                self.val_texts = val_texts_or_path
                self.dataset_length["eval"] = len(self.val_texts)

    def per_objective_log(self, split: str) -> Dict[str, float]:
        out_logs = {}
        # aggregate per-logging-steps, or per-evaluation-steps, keep the results of unprocessed evaluations
        logger.warning("Constructing %s logs based on %s samples" % (split, len(self.outputs_history[split])))
        if self.outputs_history[split]:
            # if nonempty (last evaluation)
            # aggregate recent losses into the report, clear out losses cache
            mean_loss = sum(self.loss_history[split]) / len(self.loss_history[split])
            self.evaluations_history[split]["loss"].append(mean_loss)

            out_logs["%s_%s_loss" % (split, self)] = mean_loss
            out_logs["%s_%s_num_batches" % (split, self)] = len(self.outputs_history[split])
            for evaluator in self.evaluators[split]:
                n_last_logits = [logits for logits, labels in self.outputs_history[split]]
                n_last_labels = [labels for logits, labels in self.outputs_history[split]]

                # evaluator should already return an aggregated value, so unlike loss, we don't average it
                evaluator_value = evaluator(n_last_logits, n_last_labels, self.tokenizer)
                self.evaluations_history[split][evaluator].append(evaluator_value)
                out_logs["%s_%s_%s" % (split, self, evaluator)] = evaluator_value

            # LM logits, each of shape (batch_size, n_tokens, vocab_size) can consume a lot of memory
            # we erase the raw outputs after the logging, to save space, but we remember the values of Evaluators
            self.outputs_history[split] = []
        return out_logs

    def has_converged(self, patience: int) -> bool:
        convergence_evaluators = [e for e in self.evaluators['eval']
                                  if isinstance(e, EvaluatorBase) and e.determines_convergence]
        if convergence_evaluators:
            stopping_evaluator = convergence_evaluators[0]
        else:
            stopping_evaluator = "loss"

        # the objective was not active in the recent logging interval -> it should not be marked converged
        if not any(self.evaluations_history["train"][e] for e in self.evaluators['train']):
            return False

        passed_patience_evals = len(self.evaluations_history["eval"][stopping_evaluator]) > patience
        if not passed_patience_evals:
            # less than `patience` evaluations has passed so far
            return False
        last_n = self.evaluations_history["eval"][stopping_evaluator][-patience:]
        previous = self.evaluations_history["eval"][stopping_evaluator][:-patience]
        if stopping_evaluator == "loss" or stopping_evaluator.smaller_is_better:
            did_not_improve = min(previous) <= min(last_n)
        else:
            did_not_improve = max(previous) >= max(last_n)

        if did_not_improve:
            logger.warning("Objective `%s` convergence metric `%s` did not improve for %s eval steps. History: %s" %
                           (self, stopping_evaluator, patience, last_n))

        return passed_patience_evals and did_not_improve

    def _register_outputs(self, split: str, logit_outputs: torch.FloatTensor, labels: torch.LongTensor,
                          remember_at_most: int = 100) -> None:
        self.outputs_history[split].append((logit_outputs.detach().cpu(), labels.detach().cpu()))

        # memory saving shortcut
        self.outputs_history[split] = self.outputs_history[split][-remember_at_most:]

    @abc.abstractmethod
    def _compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        pass

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor, split: str) -> torch.FloatTensor:
        self._register_outputs(split, logit_outputs, labels)
        loss = self._compute_loss(logit_outputs, labels)
        self.loss_history[split].append(loss.item())

        self.progressbar[split].set_postfix(refresh=False, split=split, loss=loss.item(), epoch=self.epoch)
        self.progressbar[split].update(1)

        return loss

    @abc.abstractmethod
    def _get_inputs_iterator(self, split: str) -> Iterable:
        pass

    def get_dataset(self, split: str, objective_i: int, device: Union[str, torch.device]) -> AdaptationDataset:
        self.epoch += 1 if split == "train" else 0

        self.progressbar[split] = trange(self.dataset_length[split] // self.batch_size,
                                         desc=str(self),
                                         unit="batches",
                                         position=objective_i,
                                         leave=True)
        self.progressbar[split].set_postfix(refresh=False, split=split, epoch=self.epoch, loss=-1)

        inputs_iter = self._get_inputs_iterator(split)

        def _sample_to_device(sample: Dict[str, torch.LongTensor]) -> Dict[str, torch.LongTensor]:
            return {k: v.to(device) if k != "oid" else v for k, v in sample.items()}

        def _add_oid(sample: Dict[str, Union[int, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            sample["oid"] = id(self)
            return sample

        device_inputs_iter = map(_sample_to_device, inputs_iter)
        device_inputs_iter = map(_add_oid, device_inputs_iter)

        return TransformerAdaptationDataset(device_inputs_iter, self.dataset_length[split])

    @abc.abstractmethod
    def _per_split_iterators(self, split: str) -> Union[Iterable[str], Tuple[Iterable[str], Iterable[str]]]:
        pass

    def register_compatible_head_model(self, lang_module: LangModule,
                                       other_objective: Optional["Objective"],
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        head_config = objective_args_for_head_config if objective_args_for_head_config is not None else {}

        if other_objective is not None:
            logger.warning("Objective %s will share %s head with %s objective",
                           self, self.compatible_head.name, other_objective)
            preloaded_module = other_objective.compatible_head_model

        return lang_module.load_training_head(self.compatible_head, str(id(self)), head_config, preloaded_module)

    def __str__(self) -> str:
        if self.given_id:
            return str("%s-%s" % (self.given_id, self.__class__.__name__))
        else:
            return self.__class__.__name__


class UnsupervisedObjective(Objective, abc.ABC):

    def _per_split_iterator_single(self, split: str) -> Iterable[str]:
        if split == "train":
            if self.texts is not None:
                sources_iter = iter(self.texts)
            else:
                sources_iter = AdaptationDataset.iter_text_file_per_line(self.texts_path)
        elif split == "eval":
            assert self.val_texts is not None, "Objective %s did not get any validation texts :( " \
                                               "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, " \
                                               "or set Objective(val_texts) param." % self

            if self.texts is not None:
                sources_iter = iter(self.val_texts)
            else:
                sources_iter = AdaptationDataset.iter_text_file_per_line(self.val_texts_path)
        else:
            raise ValueError("Unrecognized split: %s" % split)

        return sources_iter

    def _per_split_iterators(self, split: str) -> Tuple[Iterable[str], Iterable[str]]:
        return self._per_split_iterator_single(split), self._per_split_iterator_single(split)


class SupervisedObjective(UnsupervisedObjective, abc.ABC):

    labels_path: Optional[str] = None
    labels: Optional[List[str]] = None

    val_labels_path: Optional[str] = None
    val_labels: Optional[List[str]] = None

    labels_map: Dict[str, int] = {}

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 labels_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None,
                 train_evaluators: Sequence[EvaluatorBase] = (),
                 val_evaluators: Sequence[EvaluatorBase] = (),
                 share_other_objective_head: Optional["Objective"] = None,
                 objective_module: Optional[torch.nn.Module] = None,
                 objective_id: Optional[str] = ""):

        if type(labels_or_path) == str:
            self.labels_path = labels_or_path
        else:
            self.labels = labels_or_path

        if val_labels_or_path is not None:
            if type(val_labels_or_path) == str:
                self.val_labels_path = val_labels_or_path
            else:
                self.val_labels = val_labels_or_path

        # init will call register_compatible_head_model, which resolves num_labels for new head config from self.labels
        super().__init__(lang_module=lang_module,
                         batch_size=batch_size,
                         texts_or_path=texts_or_path,
                         val_texts_or_path=val_texts_or_path,
                         train_evaluators=train_evaluators,
                         val_evaluators=val_evaluators,
                         share_other_objective_head=share_other_objective_head,
                         objective_module=objective_module,
                         objective_id=objective_id)

    def register_compatible_head_model(self, lang_module: LangModule,
                                       other_objective: Optional["Objective"],
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        if self.labels is not None:
            all_labels = self.labels
        else:
            with open(self.labels_path) as f:
                all_labels = [l.strip() for l in f.readlines()]
        if self.compatible_head == Head.TOKEN_CLASSIFICATION:
            all_labels = set(itertools.chain(*(token_labels_str.split() for token_labels_str in all_labels)))

        self.labels_map = {val: i for i, val in enumerate(sorted(all_labels))}

        objective_args_for_head_config = {**objective_args_for_head_config,
                                          "num_labels": len(all_labels),
                                          "label2id": self.labels_map,
                                          "id2label": {v: k for k, v in self.labels_map.items()}}
        head_module = super().register_compatible_head_model(lang_module, other_objective,
                                                             objective_args_for_head_config, preloaded_module)
        return head_module

    def _per_split_iterators(self, split: str) -> Tuple[Iterable[str], Iterable[str]]:
        sources_iter, _ = super(SupervisedObjective, self)._per_split_iterators(split)

        if split == "train":
            if self.texts is not None:
                targets_iter = iter(self.labels)
            else:
                targets_iter = AdaptationDataset.iter_text_file_per_line(self.labels_path)
        elif split == "eval":
            assert self.val_labels is not None, "Objective %s did not get any validation labels :( " \
                                                "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, " \
                                                "or set Objective(val_labels) param." % self
            if self.texts is not None:
                targets_iter = iter(self.val_labels)
            else:
                targets_iter = AdaptationDataset.iter_text_file_per_line(self.val_labels_path)
        else:
            raise ValueError("Unrecognized split: %s" % split)

        return sources_iter, targets_iter
