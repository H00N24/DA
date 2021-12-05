from domain_adaptation.evaluators.generative import BLEU
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.CLM import CausalLanguageModelingUnsup, CausalLanguageModelingSup
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
from domain_adaptation.objectives.seq2seq_soft import Seq2SeqMinimumRiskTraining, MinimumFlow
from utils import paths, test_base_models

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"


def assert_module_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(**dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"], split)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def test_mrt_objective():
    lang_module = LangModule(test_base_models["translation"])

    metrics_args = {"additional_sep_char": "▁"}
    train_metrics = [BLEU(**metrics_args)]

    objective = Seq2SeqMinimumRiskTraining(lang_module=lang_module,
                                           texts_or_path=paths["texts"]["target_domain"]["translation"],
                                           labels_or_path=paths["labels"]["target_domain"]["translation"],
                                           batch_size=4,
                                           source_lang_id="en",
                                           target_lang_id="cs",
                                           train_evaluators=train_metrics)

    assert_module_objective_ok(lang_module, objective)


def test_minimum_flow_objective():
    lang_module = LangModule(test_base_models["translation"])

    objective = MinimumFlow(lang_module=lang_module,
                            texts_or_path=paths["texts"]["target_domain"]["translation"],
                            labels_or_path=paths["labels"]["target_domain"]["translation"],
                            batch_size=4,
                            source_lang_id="en",
                            target_lang_id="cs")

    assert_module_objective_ok(lang_module, objective)
