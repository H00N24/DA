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

            # TODO: register expected output sizes
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

    def resize_head(self, selected_head: Head, target_size: int, head_params_prefix: str = "classifier") -> None:
        raise DeprecationWarning("This is not a supported way of adjusting the model.")
        if selected_head in self.resized_heads:
            raise UserWarning("Head %s has already been resized, but I will resize it again. "
                              "Resizing re-initializes some model parameters, make sure you are not doing it too ofter")

        def nested_assign(parent_module: Union[torch.nn.Module, torch.nn.Parameter],
                          reassigned_param: str, reassignment: torch.nn.Parameter):
            if type(parent_module) != torch.nn.Parameter:
                all_params_dict = dict(parent_module.named_parameters())
                child_module = all_params_dict[reassigned_param]

                nested_assign(child_module, reassigned_param.split(".", maxsplit=2)[1], reassignment)
            else:
                setattr(parent_module, reassigned_param, reassignment)
                parent_module.shape = reassignment.shape

        with torch.no_grad():
            requested_head_model = self.trainable_models[selected_head.name]
            all_params = requested_head_model.state_dict()
            head_params = [params for params in all_params.items() if head_params_prefix in params[0]]
            prev_output_size = head_params[-1][1].shape[0]
            for name, param in head_params:
                # resize head param, if any of its dimensions match the previous output size
                # note that this is a convenient heuristic, but does not reassure the correctness
                orig_shape = param.shape
                if prev_output_size in orig_shape:
                    new_shape = [s if s != prev_output_size else target_size for s in orig_shape]
                    new_param = torch.nn.Parameter(torch.randn(new_shape), requires_grad=True)
                    nested_assign(requested_head_model, name, new_param)
                    self.resized_heads.add(selected_head)
            return

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

    def save_pretrained(self, save_dir: str, saved_head: Head) -> None:
        """ Saves a lang_module instance with given head to `save_dir`.
        TODO: save all heads at once: register all params to a single lang_module and persist the one.
        On loading, models with distinct heads might pick the corresponding params by their **non-colliding** names

        :param save_dir:
        :param saved_head:
        :return:
        """
        self.tokenizer.save_pretrained(save_dir)
        self.trainable_models[saved_head.name].save_pretrained(save_dir)
