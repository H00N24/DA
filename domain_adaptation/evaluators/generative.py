import torch
from transformers import PreTrainedTokenizer

from domain_adaptation.evaluators.evaluator_base import EvaluatorBase
from sacrebleu import corpus_bleu


class BLEU(EvaluatorBase):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)

    def __call__(self, outputs:   torch.FloatTensor, labels: torch.LongTensor):

        expected_str = self.tokenizer.batch_decode(labels)
        actual_str = self.tokenizer.batch_decode(outputs)

        return corpus_bleu(expected_str, [actual_str])
