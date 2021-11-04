import abc
from typing import List, Union, Optional, Iterable, Tuple, Dict

import torch
from tqdm import trange

from ..lang_module import LangModule
from ..utils import AdaptationDataset, Head, TransformerAdaptationDataset


class Objective(abc.ABC):

    compatible_head: Head
    epoch: int

    texts: Optional[List[str]]
    texts_path: Optional[str]

    val_texts_path: Optional[str]
    val_texts: Optional[List[str]]

    dataset_length: Dict[str, int]
    loss_history: Dict[str, List[float]]
    progressbar: Dict[str, trange]

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None):
        self.batch_size = batch_size
        self.tokenizer = lang_module.tokenizer
        self.compatible_head_model = self.pick_compatible_head_model(lang_module)

        self.epoch = 0
        self.dataset_length = {"train": 0, "eval": 0}
        self.loss_history = {"train": [], "eval": []}
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

        if val_texts_or_path is not None:
            if type(val_texts_or_path) == str:
                self.val_texts_path = val_texts_or_path
                with open(self.val_texts_path) as f:
                    self.dataset_length["eval"] = len(f.readlines())
            else:
                self.val_texts = val_texts_or_path
                self.dataset_length["eval"] = len(self.val_texts)

    @abc.abstractmethod
    def _compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        pass

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor, split: str) -> torch.FloatTensor:
        loss = self._compute_loss(logit_outputs, labels)
        self.loss_history[split].append(loss.item())

        self.progressbar[split].set_postfix(refresh=False, split=split, loss=loss.item(), epoch=self.epoch)
        self.progressbar[split].update(1)

        return loss

    @abc.abstractmethod
    def _get_inputs_iterator(self, split: str) -> Iterable:
        pass

    def get_dataset(self, split: str,
                    objective_i: int,
                    device: Union[str, torch.device],
                    epoch: int = 0) -> AdaptationDataset:
        self.epoch = epoch

        self.progressbar[split] = trange(self.dataset_length[split],
                                         desc=str(self),
                                         unit="batches",
                                         position=objective_i,
                                         leave=True)
        self.progressbar[split].set_postfix(refresh=False, split=split, epoch=epoch, loss=-1)

        inputs_iter = self._get_inputs_iterator(split)

        def _sample_to_device(sample: Dict[str, torch.LongTensor]) -> Dict[str, torch.LongTensor]:
            return {k: v.to(device) if k != "oid" else v for k, v in sample.items()}

        device_inputs_iter = map(_sample_to_device, inputs_iter)

        return TransformerAdaptationDataset(device_inputs_iter, objective_id=id(self))

    @abc.abstractmethod
    def _per_split_iterators(self, split: str) -> Union[Iterable[str], Tuple[Iterable[str], Iterable[str]]]:
        pass

    def pick_compatible_head_model(self, lang_module: LangModule) -> torch.nn.Module:
        try:
            return [module for head, module in lang_module.trainable_models.items()
                    if head == self.compatible_head.name][0]
        except IndexError:
            raise ValueError("No head of the loaded LangModule is compatible with %s objective!" % self)

    def __str__(self) -> str:
        return str(self.__class__.__name__)


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

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 labels_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, val_texts_or_path)

        if type(labels_or_path) == str:
            self.labels_path = labels_or_path
        else:
            self.labels = labels_or_path

        if val_labels_or_path is not None:
            if type(val_labels_or_path) == str:
                self.val_labels_path = val_labels_or_path
            else:
                self.val_labels = val_labels_or_path

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
