from typing import List

import torch
from transformers import PreTrainedTokenizer

from domain_adaptation.evaluators.evaluator_base import EvaluatorBase
from domain_adaptation.utils import Head


class MacroAccuracy(EvaluatorBase):

    compatible_head: Head = Head.TOKEN_CLASSIFICATION
    smaller_is_better: bool = False

    @staticmethod
    def __call__(logit_outputs: List[torch.FloatTensor],
                 labels: List[torch.LongTensor],
                 tokenizer: PreTrainedTokenizer):
        return sum(logit_outputs.argmax(-1) == labels) / labels.flatten().shape[0]

