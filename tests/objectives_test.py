from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.CLM import DecoderSequence2Sequence, CausalLanguageModelingUnsup, \
    CausalLanguageModelingSup
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.utils import Head
from utils import paths, test_base_models

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"


def assert_model_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, epoch=0)))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(objective.compatible_head, **dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"], split)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def test_token_classification_objective():
    lang_module = LangModule(test_base_models["token_classification"], head_types=[Head.TOKEN_CLASSIFICATION],
                             head_kwargs=[{"num_labels": 3}])
    objective = TokenClassification(lang_module,
                                    texts_or_path=paths["texts"]["target_domain"]["ner"],
                                    labels_or_path=paths["labels"]["target_domain"]["ner"],
                                    batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def test_mlm_objective():
    lang_module = LangModule(test_base_models["token_classification"], head_types=[Head.LANGUAGE_MODEL])
    objective = MaskedLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                       batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def test_clm_unsup_objective():
    lang_module = LangModule(test_base_models["token_classification"], head_types=[Head.LANGUAGE_MODEL])
    objective = CausalLanguageModelingUnsup(lang_module,
                                            texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                            batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def test_clm_sup_objective():
    lang_module = LangModule(test_base_models["translation"], head_types=[Head.LANGUAGE_MODEL])
    objective = CausalLanguageModelingSup(lang_module,
                                          texts_or_path=paths["texts"]["target_domain"]["translation"],
                                          labels_or_path=paths["labels"]["target_domain"]["translation"],
                                          source_lang_id="en",
                                          target_lang_id="cs",
                                          batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def test_denoising_objective():
    lang_module = LangModule(test_base_models["token_classification"], head_types=[Head.LANGUAGE_MODEL])
    objective = DenoisingObjective(lang_module, texts_or_path=paths["texts"]["target_domain"]["unsup"], batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def test_supervised_seq2seq_objective():
    lang_module = LangModule(test_base_models["translation"], head_types=[Head.LANGUAGE_MODEL])
    objective = DecoderSequence2Sequence(lang_module,
                                         texts_or_path=paths["texts"]["target_domain"]["translation"],
                                         labels_or_path=paths["labels"]["target_domain"]["translation"],
                                         batch_size=4,
                                         source_lang_id="en",
                                         target_lang_id="cs")

    assert_model_objective_ok(lang_module, objective)

# run as standalone scripts:

# token_classification_objective()
# mlm_objective()
# test_denoising_objective()
# test_supervised_seq2seq_objective()
