from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.utils import Head

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"


def assert_model_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split)))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(objective.compatible_head, **dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"])

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def token_classification_objective():
    lang_module = LangModule("bert-base-multilingual-cased", head_types=[Head.TOKEN_CLASSIFICATION],
                             head_kwargs=[{"num_labels": 3}])
    objective = TokenClassification(lang_module,
                                    texts_or_path=sup_target_domain_texts,
                                    labels_or_path=sup_target_domain_labels,
                                    batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def mlm_objective():
    lang_module = LangModule("bert-base-multilingual-cased", head_types=[Head.LANGUAGE_MODEL])
    objective = MaskedLanguageModeling(lang_module, texts_or_path=unsup_target_domain_texts, batch_size=4)

    assert_model_objective_ok(lang_module, objective)


def seq2seq_objective():
    lang_module = LangModule("bert-base-multilingual-cased", head_types=[Head.LANGUAGE_MODEL])
    objective = DenoisingObjective(lang_module, texts_or_path=unsup_target_domain_texts, batch_size=4)

    assert_model_objective_ok(lang_module, objective)


# OK:
token_classification_objective()

# OK:
mlm_objective()

# OK:
seq2seq_objective()
