import abc

import torch
from transformers import PreTrainedTokenizer


class EvaluatorBase(abc.ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def __call__(self, outputs: torch.FloatTensor, labels: torch.LongTensor):
        pass
