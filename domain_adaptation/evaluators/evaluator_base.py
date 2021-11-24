import abc
from typing import List

import torch
from transformers import PreTrainedTokenizer

from domain_adaptation.utils import Head


class EvaluatorBase(abc.ABC):

    compatible_head: Head
    smaller_is_better: bool

    def __init__(self, decides_convergence: bool = False):
        self.determines_convergence = decides_convergence

    @staticmethod
    @abc.abstractmethod
    def __call__(logit_outputs: List[torch.FloatTensor],
                 labels: List[torch.LongTensor],
                 tokenizer: PreTrainedTokenizer):
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__)
