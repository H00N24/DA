from typing import List, Union, Optional, Iterator

import torch
from torch.nn import CrossEntropyLoss
from transformers import default_data_collator, DataCollatorForSeq2Seq

from ..lang_module import LangModule
from ..objectives.objective_base import SupervisedObjective, Sequence2SequenceMixin, UnsupervisedObjective, \
    DecoderSequence2SequenceMixin


class CausalLanguageModelingUnsup(UnsupervisedObjective, Sequence2SequenceMixin):

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, val_texts_or_path)

        self.collator = default_data_collator

    def _compute_loss(self, lm_logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # transformers.RobertaForCausalLM.forward()
        lm_logit_outputs = lm_logit_outputs[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(lm_logit_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        return lm_loss

    def _get_inputs_iterator(self, split: str) -> Iterator:
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        collated_iter = self._get_seq2seq_collated_iterator(source_texts_iter, target_texts_iter)

        return collated_iter


class CausalLanguageModelingSup(CausalLanguageModelingUnsup, SupervisedObjective):

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 labels_or_path: Union[str, List[str]],
                 source_lang_id: str, target_lang_id: str,
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, val_texts_or_path, labels_or_path, val_labels_or_path)

        self.tokenizer.src_lang = source_lang_id
        self.tokenizer.tgt_lang = target_lang_id


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

    def _get_inputs_iterator(self, split: str) -> Iterator:
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        collated_iter = self._get_seq2seq_collated_iterator(source_texts_iter, target_texts_iter)

        return collated_iter
