from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.CLM import CausalLanguageModelingUnsup, CausalLanguageModelingSup
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
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


def test_token_classification_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = TokenClassification(lang_module,
                                    texts_or_path=paths["texts"]["target_domain"]["ner"],
                                    labels_or_path=paths["labels"]["target_domain"]["ner"],
                                    batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_mlm_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = MaskedLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_unsup_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = CausalLanguageModelingUnsup(lang_module,
                                            texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                            batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_sup_objective():
    lang_module = LangModule(test_base_models["translation"])
    objective = CausalLanguageModelingSup(lang_module,
                                          texts_or_path=paths["texts"]["target_domain"]["translation"],
                                          labels_or_path=paths["labels"]["target_domain"]["translation"],
                                          source_lang_id="en",
                                          target_lang_id="cs",
                                          batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_denoising_objective():
    lang_module = LangModule(test_base_models["translation"])
    objective = DenoisingObjective(lang_module, texts_or_path=paths["texts"]["target_domain"]["unsup"], batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_supervised_seq2seq_objective():
    lang_module = LangModule(test_base_models["translation"])
    objective = DecoderSequence2Sequence(lang_module,
                                         texts_or_path=paths["texts"]["target_domain"]["translation"],
                                         labels_or_path=paths["labels"]["target_domain"]["translation"],
                                         batch_size=4,
                                         source_lang_id="en",
                                         target_lang_id="cs")

    assert_module_objective_ok(lang_module, objective)
