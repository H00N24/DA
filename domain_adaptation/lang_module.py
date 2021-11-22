import logging

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, AutoModelWithLMHead
from typing import List, Dict, Union, Any, Optional

from .utils import Head


logger = logging.getLogger()


class LangModule(torch.nn.Module):
    """
    Module containing all the reloaded submodules for translation.
    """

    tokenizer: PreTrainedTokenizer
    # ModuleDict reassures a registration of its values by backward hooks
    trainable_models: torch.nn.ModuleDict
    heads_output_sizes: Dict[str, int] = {}

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # head_kwargs = head_kwargs if head_kwargs is not None else [{}] * len(head_types)
        # self._load_pretrained_with_heads(model_name_or_path, head_types, head_kwargs)
        self.trainable_models = torch.nn.ModuleDict()

    def load_new_head(self, objective_id: str, head_type: Head,
                      head_kwargs: Dict[str, Any], new_head: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        # manually-initialized head chosen for this objective will also be merged with other objectives and registered
        if new_head is None:
            if head_type == Head.SEQ_CLASSIFICATION:
                new_head = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, **head_kwargs)
            elif head_type == Head.TOKEN_CLASSIFICATION:
                new_head = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path, **head_kwargs)
            elif head_type == Head.LANGUAGE_MODEL:
                new_head = AutoModelWithLMHead.from_pretrained(self.model_name_or_path, **head_kwargs)
            else:
                new_head = torch.load(self.model_name_or_path, **head_kwargs)

        # this applies to the 2nd+ -added models: they adopt the shared parameters of the first lang_module
        if len(self.trainable_models) > 1:
            unmatched_modules = self._partially_match_models(list(self.trainable_models.values())[0],
                                                             self.trainable_models[head_type.name])
            # this can contain a deep stack of layers, hence in general, it can not be checked automatically
            logger.warning("These layers of the loaded %s were not merged: %s" % (head_type.name, unmatched_modules))
        self.trainable_models[objective_id] = new_head

        return new_head

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

    def forward(self, **inputs) -> torch.LongTensor:
        selected_head_model = self.trainable_models[str(inputs["oid"])]

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
