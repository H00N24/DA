import abc
from typing import List, Union, Optional, Iterable, Dict, Iterator, Callable

import torch
from transformers import DataCollatorForSeq2Seq

from ..lang_module import LangModule
from ..objectives.objective_base import SupervisedObjective, Objective
from ..utils import Head


class Sequence2SequenceMixin(Objective, abc.ABC):

    compatible_head: Head = Head.LANGUAGE_MODEL
    collator: Callable[[List[Dict[str, torch.FloatTensor]]], List[Dict[str, torch.FloatTensor]]]

    def _get_seq2seq_collated_iterator(self, source_texts: Iterable[str], target_texts: Iterable[str]) -> Iterator:
        features_batch = []

        for source_text, target_text in zip(source_texts, target_texts):

            sample_features = self.tokenizer(source_text, truncation=True)
            sample_targets = self.tokenizer(target_text, truncation=True)

            features_batch.append({"input_ids": sample_features.input_ids,
                                   "attention_mask": sample_features.attention_mask,
                                   "labels": sample_targets.input_ids})
            if len(features_batch) == self.batch_size:
                yield self.collator(features_batch)
                features_batch = []

        if features_batch:
            # yield last nonempty residual batch
            yield self.collator(features_batch)

    def _compute_loss(self, lm_logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(lm_logit_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))

        return lm_loss

    def _get_inputs_iterator(self, split: str) -> Iterator:
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        collated_iter = self._get_seq2seq_collated_iterator(source_texts_iter, target_texts_iter)

        return collated_iter


class DecoderSequence2SequenceMixin(Sequence2SequenceMixin, abc.ABC):

    def pick_compatible_head_model(self, lang_module: LangModule) -> torch.nn.Module:
        try:
            return [module for head, module in lang_module.trainable_models.items()
                    if head == self.compatible_head.name
                    and hasattr(module, "prepare_decoder_input_ids_from_labels")][0]
        except IndexError:
            raise ValueError("No head of the loaded LangModule is compatible with %s objective!"
                             "\nNote that the module compatible with DecoderSequence2SequenceMixin"
                             "\nmust have `prepare_decoder_input_ids_from_labels` method, see e.g."
                             "\ntransformers.BartModel." % self)


class DecoderSequence2Sequence(DecoderSequence2SequenceMixin, SupervisedObjective):

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 labels_or_path: Union[str, List[str]],
                 source_lang_id: str, target_lang_id: str,
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, labels_or_path, val_texts_or_path, val_labels_or_path)

        self.tokenizer.src_lang = source_lang_id
        self.tokenizer.tgt_lang = target_lang_id

        # data collator and loss is the only difference to CLM objectives
        self.collator = DataCollatorForSeq2Seq(lang_module.tokenizer)
