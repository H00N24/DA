from typing import List, Dict, Iterable, Tuple, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForTokenClassification

from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.objective_base import SupervisedObjective
from domain_adaptation.utils import AdaptationDataset, TransformerAdaptationDataset, Head


class TokenClassification(SupervisedObjective):

    compatible_head = Head.TOKEN_CLASSIFICATION

    # TODO: this should be persisted in config of *_classification-heads models
    labels_map: Dict[str, int] = {}

    def __init__(self, lang_module: LangModule, batch_size: int,
                 texts_or_path: Union[str, List[str]], labels_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None):
        super().__init__(lang_module, batch_size, texts_or_path, labels_or_path, val_texts_or_path, val_labels_or_path)

        self.collator = DataCollatorForTokenClassification(lang_module.tokenizer)

    def _wordpiece_token_label_alignment(self, texts: Iterable[str],
                                         labels: Iterable[str]) -> Iterable[Dict[str, torch.LongTensor]]:
        batch_features = []

        for text, text_labels in zip(texts, labels):
            tokens = text.split()
            labels = text_labels.split()

            tokenizer_encodings = self.tokenizer(text)
            # attention mask is lang_module-specific
            attention_mask = tokenizer_encodings.attention_mask
            wpiece_ids = tokenizer_encodings.input_ids
            wordpieces = self.tokenizer.batch_decode(wpiece_ids)

            out_label_ids = []

            # next token lookup - avoid out-of-index, and exclude from token labels
            tokens.append(wordpieces[-1])
            labels.append("O")

            assert len(tokens) == len(labels), \
                "A number of tokens in the first line is different than a number of labels. " \
                "Text: %s \nLabels: %s" % (text, text_labels)

            # assign current label to current wordpiece until the current_token is fully iterated-over
            current_token = tokens.pop(0)
            current_label = labels.pop(0)
            for wpiece_id, wpiece in zip(wpiece_ids, wordpieces):
                next_token = tokens[0]
                if next_token.startswith(wpiece):
                    # if the next token starts with a current wordpiece, move to the next token + label
                    current_token = tokens.pop(0)
                    current_label = labels.pop(0)
                try:
                    out_label_ids.append(self.labels_map[current_label])
                except KeyError:
                    # previously-unseen label
                    self.labels_map[current_label] = len(self.labels_map)
                    out_label_ids.append(self.labels_map[current_label])

            batch_features.append({"input_ids": wpiece_ids,
                                   "attention_mask": attention_mask,
                                   "labels": out_label_ids})
            # maybe yield a batch
            if len(batch_features) == self.batch_size:
                yield self.collator(batch_features)
                batch_features = []
        if batch_features:
            yield self.collator(batch_features)

        # check that the number of outputs of the selected compatible head matches the just-parsed data set
        num_outputs = list(self.compatible_head_model.parameters())[-1].shape[0]
        num_labels = len(self.labels_map)
        assert num_outputs == num_labels, "A number of the outputs for the selected %s head (%s) " \
                                          "does not match a number of token labels (%s)" \
                                          % (self.compatible_head, num_outputs, num_labels)

    def get_dataset(self, split: str) -> TransformerAdaptationDataset:

        if self.texts is not None:
            texts_iter = iter(self.texts)
            labels_iter = iter(self.labels)
        else:
            texts_iter = AdaptationDataset.iter_text_file_per_line(self.texts_path)
            labels_iter = AdaptationDataset.iter_text_file_per_line(self.labels_path)

        aligned_collated_iter = self._wordpiece_token_label_alignment(texts_iter, labels_iter)

        return TransformerAdaptationDataset(aligned_collated_iter, objective_id=id(self))

    def compute_loss(self, logit_outputs: torch.FloatTensor,
                     labels: torch.LongTensor,
                     attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        # generic token classification loss, originally implemented e.g. in transformers.BertForTokenClassification

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logit_outputs.view(-1, len(self.labels_map))
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logit_outputs.view(-1, len(self.labels_map)), labels.view(-1))

        return loss


class SequenceClassification(SupervisedObjective):

    compatible_head = Head.SEQ_CLASSIFICATION

    def get_dataset(self, split: str) -> AdaptationDataset:
        raise NotImplementedError()

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError()
