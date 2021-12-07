from typing import List, Union, Optional, Sequence

import torch
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq

from .seq2seq import Sequence2SequenceMixin
from ..evaluators.evaluator_base import EvaluatorBase
from ..lang_module import LangModule
from ..objectives.objective_base import SupervisedObjective, UnsupervisedObjective


class CLMMixin:

    @staticmethod
    def _compute_loss(lm_logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # from transformers.GPT2LMHeadModel.forward()
        shift_logits = lm_logit_outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class CausalLanguageModelingUnsup(UnsupervisedObjective, Sequence2SequenceMixin, CLMMixin):

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 train_evaluators: Sequence[EvaluatorBase] = (),
                 val_evaluators: Sequence[EvaluatorBase] = (),
                 share_other_objective_head: Optional["Objective"] = None,
                 objective_module: Optional[torch.nn.Module] = None,
                 objective_id: Optional[str] = ""):
        super().__init__(lang_module=lang_module,
                         batch_size=batch_size,
                         texts_or_path=texts_or_path,
                         val_texts_or_path=val_texts_or_path,
                         train_evaluators=train_evaluators,
                         val_evaluators=val_evaluators,
                         share_other_objective_head=share_other_objective_head,
                         objective_module=objective_module,
                         objective_id=objective_id)

        # TODO: check that CLM with seq2seq collator does not see forward
        self.collator = DataCollatorForSeq2Seq(lang_module.tokenizer)


class CausalLanguageModelingSup(SupervisedObjective, Sequence2SequenceMixin, CLMMixin):

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 labels_or_path: Union[str, List[str]],
                 source_lang_id: str,
                 target_lang_id: str,
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 val_labels_or_path: Optional[Union[str, List[str]]] = None,
                 train_evaluators: Sequence[EvaluatorBase] = (),
                 val_evaluators: Sequence[EvaluatorBase] = (),
                 share_other_objective_head: Optional["Objective"] = None,
                 objective_module: Optional[torch.nn.Module] = None,
                 objective_id: Optional[str] = ""):
        super().__init__(lang_module=lang_module, batch_size=batch_size, texts_or_path=texts_or_path,
                         labels_or_path=labels_or_path, val_texts_or_path=val_texts_or_path,
                         val_labels_or_path=val_labels_or_path, train_evaluators=train_evaluators,
                         val_evaluators=val_evaluators, share_other_objective_head=share_other_objective_head,
                         objective_module=objective_module, objective_id=objective_id)

        self.tokenizer.src_lang = source_lang_id
        self.tokenizer.tgt_lang = target_lang_id

        self.collator = DataCollatorForSeq2Seq(lang_module.tokenizer)

