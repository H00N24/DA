from typing import List, Union, Iterable, Dict, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.objective_base import UnsupervisedObjective
from domain_adaptation.utils import AdaptationDataset, TransformerAdaptationDataset, Head


class MaskedLanguageModeling(UnsupervisedObjective):
    """
    TODO: a support for pre-training will require an option of initialization without pre-trained tokenizer and lang_module
    https://github.com/huggingface/transformers/blob/5e3b4a70d3d17f2482d50aea230f7ed42b3a8fd0
    /examples/pytorch/language-modeling/run_mlm.py#L357
    """
    compatible_head = Head.LANGUAGE_MODEL

    def __init__(self, lang_module: LangModule, batch_size: int, texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 masking_application_prob: float = 0.15,
                 full_token_masking: bool = False):
        super().__init__(lang_module, batch_size, texts_or_path, val_texts_or_path)
        if full_token_masking:
            self.collator = DataCollatorForWholeWordMask(lang_module.tokenizer,
                                                         mlm_probability=masking_application_prob)
        else:
            self.collator = DataCollatorForLanguageModeling(lang_module.tokenizer,
                                                            mlm_probability=masking_application_prob)

    def _mask_some_tokens(self, texts: Iterable[str]) -> Iterable[Dict[str, torch.LongTensor]]:
        batch_features = []
        for text in texts:
            input_features = self.tokenizer(text)
            batch_features.append(input_features)

            # maybe yield a batch
            if len(batch_features) == self.batch_size:
                # selection of masked tokens, padding and labeling is provided by transformers.DataCollatorForLM
                yield self.collator(batch_features)
                batch_features = []
        # yield remaining texts in collected batch
        if batch_features:
            yield self.collator(batch_features)

    def get_dataset(self, split: str) -> TransformerAdaptationDataset:
        if self.texts is not None:
            texts_iter = iter(self.texts)
        else:
            texts_iter = AdaptationDataset.iter_text_file_per_line(self.texts_path)

        collated_iter = self._mask_some_tokens(texts_iter)
        return TransformerAdaptationDataset(collated_iter, objective_id=id(self))

    def compute_loss(self, mlm_token_logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # generic token classification loss, can be found e.g. in transformers.BertForMaskedLM
        loss_fct = CrossEntropyLoss()
        vocab_size = mlm_token_logits.size()[-1]
        masked_lm_loss = loss_fct(mlm_token_logits.view(-1, vocab_size), labels.view(-1))
        return masked_lm_loss
