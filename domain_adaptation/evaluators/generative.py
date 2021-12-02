import abc
from bert_score import BERTScorer
import itertools
from typing import List, Sequence, Optional

import torch
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head


class GenerativeEvaluator(EvaluatorBase, abc.ABC):

    compatible_head: Head = Head.LANGUAGE_MODEL

    def __init__(self, decides_convergence: bool = False, additional_sep_char: Optional[str] = None):
        super().__init__(decides_convergence)
        self.additional_sep_char = additional_sep_char

    def __call__(self,
                 logit_outputs: List[torch.FloatTensor],
                 labels: List[torch.LongTensor],
                 tokenizer: PreTrainedTokenizer):
        for label_t in labels:
            label_t[label_t < 0] = 0

        expected_str = itertools.chain(*(tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
                                         for batch_labels in labels))
        # single-pass prediction - likely gives worse results than beam search (model.generate())
        argmax_tokens = itertools.chain(*(torch.argmax(batch_logits, -1) for batch_logits in logit_outputs))
        actual_str = tokenizer.batch_decode(argmax_tokens, skip_special_tokens=True)
        if self.additional_sep_char is not None:
            expected_str = [" ".join(expected_one.split(self.additional_sep_char)) for expected_one in expected_str]
            actual_str = [" ".join(actual_one.split(self.additional_sep_char)) for actual_one in actual_str]

        return self.evaluate_str(list(expected_str), actual_str)

    @abc.abstractmethod
    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        pass


class BLEU(GenerativeEvaluator):

    smaller_is_better: bool = False

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        return corpus_bleu(actual_list, [list(expected_list)]).score


class ROUGE(GenerativeEvaluator):

    smaller_is_better: bool = False

    def __init__(self, decides_convergence: bool = False, additional_sep_char: Optional[str] = None):
        super().__init__(decides_convergence, additional_sep_char)
        self.determines_convergence = decides_convergence
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        all_scores = [self.scorer.score(expected, actual)['rougeL'].recall
                      for expected, actual in zip(expected_list, actual_list)]
        return sum(all_scores) / len(expected_list)


class BERTScore(GenerativeEvaluator):

    def __init__(self, decides_convergence: bool = False, additional_sep_char: Optional[str] = None):
        super().__init__(decides_convergence, additional_sep_char)

        self.determines_convergence = decides_convergence
        self.scorer = BERTScorer(lang="any", model_type="bert-base-multilingual-cased")

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        b_prec, b_rec, b_f_scores = self.scorer.score(expected_list, actual_list)
        return b_f_scores.mean().cpu().item()
