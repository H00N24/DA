import itertools
from typing import List, Sequence

import torch
from sacrebleu import corpus_bleu
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head


class BLEU(EvaluatorBase):

    compatible_head: Head = Head.LANGUAGE_MODEL

    @staticmethod
    def __call__(logit_outputs: List[torch.FloatTensor],
                 labels: List[torch.LongTensor],
                 tokenizer: PreTrainedTokenizer):

        expected_str = itertools.chain(*(tokenizer.batch_decode(batch_labels) for batch_labels in labels))

        # single-pass prediction - likely gives worse results than beam search (model.generate())
        argmax_tokens = itertools.chain(*(torch.argmax(batch_logits, -1) for batch_logits in logit_outputs))
        actual_str = tokenizer.batch_decode(argmax_tokens)

        return BLEU.evaluate_str(list(expected_str), actual_str)

    @staticmethod
    def evaluate_str(expected_str: Sequence[str], actual_str: Sequence[str]) -> float:
        return corpus_bleu(actual_str, [list(expected_str)]).score
