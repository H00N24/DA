from domain_adaptation.adapter import Adapter
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.CLM import CausalDecoderLanguageModelingSup
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.schedules import SequentialSchedule, StridedSchedule
from domain_adaptation.utils import Head
from tests.utils import training_arguments, paths

unsup_target_domain_texts = paths["texts"]["target_domain"]["unsup"]
sup_target_domain_texts = paths["texts"]["target_domain"]["ner"]
sup_target_domain_labels = paths["labels"]["target_domain"]["ner"]


def test_adaptation_ner():
    # 1. pick the models - if they have the appropriate heads pre-trained, load them
    lang_module = LangModule("bert-base-multilingual-cased",
                             head_types=[Head.LANGUAGE_MODEL, Head.TOKEN_CLASSIFICATION])

    # 3. pick objectives
    # for datasets, pass in pre-loaded List[str] for in-memory iteration, or str of file path for online retrieval
    objectives = [MaskedLanguageModeling(lang_module, batch_size=1, texts_or_path=unsup_target_domain_texts),
                  TokenClassification(lang_module,
                                      batch_size=1,
                                      texts_or_path=sup_target_domain_texts,
                                      labels_or_path=sup_target_domain_labels)]

    # 4. pick a schedule of the selected objectives
    schedule = SequentialSchedule(objectives, training_arguments)

    # 5. train using Adapter
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 6. save the trained (multi-head) lang_module
    adapter.save_model("entity_detector_model")


def test_adaptation_translation():
    # 1. pick the models - if they have the appropriate heads pre-trained, load them
    lang_module = LangModule("bert-base-multilingual-cased",
                             head_types=[Head.LANGUAGE_MODEL])
    # 2. pick objectives
    objectives = [DenoisingObjective(lang_module, batch_size=1, texts_or_path=unsup_target_domain_texts), 
                  CausalDecoderLanguageModelingSup(lang_module,
                                                   batch_size=1,
                                                   texts_or_path=unsup_target_domain_texts,
                                                   labels_or_path=sup_target_domain_labels,
                                                   source_lang_id="cs", target_lang_id="en")]

    # 3. pick a schedule of the selected objectives
    schedule = StridedSchedule(objectives, training_arguments)

    # 4. train using Adapter
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 5. save the trained (multi-headed) lang_module
    adapter.save_model("translator_model")

