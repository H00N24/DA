from typing import List, Union, Optional

from ..lang_module import LangModule
from ..objectives.objective_base import UnsupervisedObjective, SupervisedObjective, LanguageModelingMixin
from ..utils import AdaptationDataset, TransformerAdaptationDataset


class CausalLanguageModelingUnsup(LanguageModelingMixin, UnsupervisedObjective):

    def get_dataset(self, split: str) -> AdaptationDataset:
        # TODO: GPT objective: for targets, move sources one-token to the right and permute the attention masks (?)
        raise NotImplementedError()


class CausalDecoderLanguageModelingSup(SupervisedObjective, LanguageModelingMixin):

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]], labels_or_path: Union[str, List[str]],
                 source_lang_id: str, target_lang_id: str,
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, labels_or_path, val_texts_or_path, val_labels_or_path)

        self.tokenizer.src_lang = source_lang_id
        self.tokenizer.tgt_lang = target_lang_id

    def get_dataset(self, split: str) -> AdaptationDataset:
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        collated_iter = self._pad_collate_inputs(source_texts_iter, target_texts_iter)

        return TransformerAdaptationDataset(collated_iter, objective_id=id(self))
