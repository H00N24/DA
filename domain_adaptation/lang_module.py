import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, AutoModelWithLMHead
from typing import List, Dict, Union, Any, Optional

from .utils import Head


class LangModule(torch.nn.Module):
    """
    Module containing all the reloaded submodules for translation.
    """

    tokenizer: PreTrainedTokenizer
    # ModuleDict reassures a registration of its values by backward hooks
    trainable_models = torch.nn.ModuleDict()
    heads_output_sizes: Dict[str, int] = {}

    def __init__(self, model_name_or_path: str, head_types: List[Head],
                 head_kwargs: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        head_kwargs = head_kwargs if head_kwargs is not None else [{}] * len(head_types)

        self._load_pretrained_with_heads(model_name_or_path, head_types, head_kwargs)

    def _load_pretrained_with_heads(self, model_name_or_path, head_types: List[Head],
                                    head_kwargs: List[Dict[str, Any]]) -> None:
        assert len(head_types) == len(head_kwargs), "A number of head arguments is different than a number of heads."
        for head_type, head_kwarg in zip(head_types, head_kwargs):
            head_str = head_type.name
            if head_type == Head.SEQ_CLASSIFICATION:
                self.trainable_models[head_str] = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                                                     **head_kwarg)
            elif head_type == Head.TOKEN_CLASSIFICATION:
                self.trainable_models[head_str] = AutoModelForTokenClassification.from_pretrained(model_name_or_path,
                                                                                                  **head_kwarg)
            elif head_type == Head.LANGUAGE_MODEL:
                self.trainable_models[head_str] = AutoModelWithLMHead.from_pretrained(model_name_or_path, **head_kwarg)
            else:
                self.trainable_models[head_str] = torch.load(model_name_or_path, **head_kwarg)

            # this applies to the <2nd-added models: they adopt the shared parameters of the first lang_module
            if len(self.trainable_models) > 1:
                unmatched_modules = self._partially_match_models(self.trainable_models[head_types[0].name],
                                                                 self.trainable_models[head_str])
                # this can contain a deep stack of layers, hence in general, it can not be checked automatically
                print("These layers of the laded %s were not merged: %s" % (head_str, unmatched_modules))

            # TODO: register expected output sizes?
            last_layer = list(self.trainable_models[head_str].parameters())[-1]
            self.heads_output_sizes[head_str] = last_layer.shape[-1]

    @staticmethod
    def _partially_match_models(orig_model: torch.nn.Module,
                                new_model: torch.nn.Module,
                                do_merge: bool = True) -> List[str]:
        """ Matches and possibly merges shared parameters of the models.
        Presumes that a vocabulary (tokenizer) of the both models does match (assured by shared self.tokenizer).
        :param orig_model: lang_module to merge to
        :param new_model: lang_module to merge
        :return: unmatched submodules
        """
        unmatched_modules = []

        for orig_param_key, orig_model_param in orig_model.named_parameters():
            # match by the param name first, then by the weights shapes and values
            if orig_param_key in dict(new_model.named_parameters()):
                new_model_param = new_model.get_parameter(orig_param_key)
                if do_merge and torch.all(orig_model_param == new_model_param):
                    new_model_param = orig_model_param
                    assert id(new_model_param) == id(orig_model_param)
                else:
                    raise ValueError("Shared submodules of the merged models have different weights! Not cool.")
            else:
                unmatched_modules.append(orig_param_key)

        return unmatched_modules

    def _head_by_labels(self, labels: torch.LongTensor) -> Head:
        if len(labels.shape) == 1:
            # sequence classification task
            return Head.SEQ_CLASSIFICATION
        elif Head.TOKEN_CLASSIFICATION.name in self.trainable_models and \
                labels.max() <= self.heads_output_sizes[Head.TOKEN_CLASSIFICATION.name]:
            # token-level task
            return Head.TOKEN_CLASSIFICATION
        elif Head.LANGUAGE_MODEL.name in self.trainable_models:
            return Head.LANGUAGE_MODEL
        else:
            raise ValueError("I did now find a compatible head for the given labels. "
                             "Automatic head inference can be omitted by selecting `requested_head` to lang_module()")

    def forward(self, requested_head: Head = None, **inputs) -> torch.LongTensor:
        label_key = "label" if "label" in inputs else "labels"

        if requested_head is None:
            if "label" not in inputs and "labels" not in inputs:
                raise AttributeError("Please give me either a head you want to infer with, or labels for training.")
            labels_tensor = inputs[label_key]
            requested_head = self._head_by_labels(labels_tensor)

        selected_head_model = self.trainable_models[requested_head.name]

        # including labels cause the loss to be computed twice - by objective + by HF models forward()
        # but labels are also used to infer decoder_input_ids of some models, so we need to pass it
        selected_head_output = selected_head_model(**{k: v for k, v in inputs.items() if k not in ("oid",)})
        # HF models produce special Output objects instead of a raw output
        logits = selected_head_output.logits if hasattr(selected_head_output, "logits") else selected_head_output

        return logits

    def reinitialize(self, head: Optional[Head] = None, seed: int = 42) -> None:
        def reinit_model_weights(m: torch.nn.Module):
            if hasattr(m, "children"):
                for m_child in m.children():
                    if hasattr(m_child, "reset_parameters"):
                        m_child.reset_parameters()
                    reinit_model_weights(m_child)

        torch.manual_seed(seed)
        if head is not None:
            self.trainable_models[head.name].apply(reinit_model_weights)
        else:
            for head, head_model in self.trainable_models.items():
                head_model.apply(reinit_model_weights)
