import abc
from typing import List, Union, Optional, Iterable, Tuple, Dict

import torch
from transformers import DataCollatorForSeq2Seq

from ..lang_module import LangModule
from ..utils import AdaptationDataset, Head


class Objective(abc.ABC):

    compatible_head: Head

    texts: Optional[List[str]] = None
    texts_path: Optional[str] = None

    val_texts_path: Optional[str] = None
    val_texts: Optional[List[str]] = None

    # TODO: propagate length properly - not train length everywhere!
    dataset_length: Dict[str, int] = {"train": 0, "eval": 0}

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]], val_texts_or_path: Optional[Union[str, List[str]]] = None):
        self.batch_size = batch_size
        self.tokenizer = lang_module.tokenizer
        self.compatible_head_model = self.pick_compatible_head_model(lang_module)

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
                self.val_texts_path = texts_or_path
                with open(self.val_texts_path) as f:
                    self.dataset_length["eval"] = len(f.readlines())
            else:
                self.val_texts = texts_or_path
                self.dataset_length["eval"] = len(self.val_texts)

    @abc.abstractmethod
    def compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abc.abstractmethod
    def _per_split_iterators(self, split: str) -> Union[Iterable[str], Tuple[Iterable[str], Iterable[str]]]:
        pass

    @abc.abstractmethod
    def get_dataset(self, split: str) -> AdaptationDataset:
        pass

    def pick_compatible_head_model(self, lang_module: LangModule) -> torch.nn.Module:
        try:
            return [module for head, module in lang_module.trainable_models.items()
                    if head == self.compatible_head.name][0]
        except IndexError:
            raise ValueError("No head of the loaded LangModule is compatible with %s objective!" % self)

    def __str__(self) -> str:
        return str(self.__class__)


class UnsupervisedObjective(Objective, abc.ABC):

    def _per_split_iterators(self, split: str) -> Iterable[str]:
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


class SupervisedObjective(UnsupervisedObjective, abc.ABC):

    labels_path: Optional[str] = None
    labels: Optional[List[str]] = None

    val_labels_path: Optional[str] = None
    val_labels: Optional[List[str]] = None

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]], labels_or_path: Union[str, List[str]],
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
        sources_iter = super(SupervisedObjective, self)._per_split_iterators(split)

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


class LanguageModelingMixin(Objective, abc.ABC):

    compatible_head: Head = Head.LANGUAGE_MODEL

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]], val_texts_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, val_texts_or_path)

        self.collator = DataCollatorForSeq2Seq(lang_module.tokenizer)

    def _pad_collate_inputs(self, source_texts: Iterable[str], target_texts: Iterable[str]):
        features_batch = []

        for source_text, target_text in zip(source_texts, target_texts):

            sample_features = self.tokenizer(source_text)
            sample_targets = self.tokenizer(target_text)

            features_batch.append({"input_ids": sample_features.input_ids,
                                   "attention_mask": sample_features.attention_mask,
                                   "labels": sample_targets.input_ids})
            if len(features_batch) == self.batch_size:
                yield self.collator(features_batch)
                features_batch = []

        if features_batch:
            # yield last nonempty residual batch
            yield self.collator(features_batch)

    def compute_loss(self, lm_logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(lm_logit_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))

        return lm_loss
