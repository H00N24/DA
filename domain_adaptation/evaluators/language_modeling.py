import abc

import torch

from domain_adaptation.evaluators.evaluator_base import EvaluatorBase
from domain_adaptation.utils import Head


class Perplexity(EvaluatorBase):

    compatible_head: Head = Head.LANGUAGE_MODEL

    @abc.abstractmethod
    def __call__(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor):
        return sum(logit_outputs.argmax(-1) == labels) / labels.flatten().shape[0]
