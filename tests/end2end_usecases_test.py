from transformers import AutoModelForSeq2SeqLM

from domain_adaptation.adapter import Adapter
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
from domain_adaptation.schedules import StridedSchedule
from domain_adaptation.utils import Head
from utils import training_arguments, paths, test_base_models

unsup_target_domain_texts = paths["texts"]["target_domain"]["unsup"]
sup_target_domain_texts = paths["texts"]["target_domain"]["ner"]
sup_target_domain_labels = paths["labels"]["target_domain"]["ner"]


def test_adaptation_ner():
    # 1. pick the models - randomly pre-initialize the appropriate heads
    lang_module = LangModule(test_base_models["token_classification"],
                             head_types=[Head.LANGUAGE_MODEL, Head.TOKEN_CLASSIFICATION],
                             head_kwargs=[{}, {"num_labels": 3}])

    # 2. pick objectives
    # for datasets, pass in pre-loaded List[str] for in-memory iteration, or str of file path for online retrieval
    objectives = [MaskedLanguageModeling(lang_module,
                                         batch_size=1,
                                         texts_or_path=paths["texts"]["target_domain"]["unsup"]),
                  TokenClassification(lang_module,
                                      batch_size=1,
                                      texts_or_path=paths["texts"]["target_domain"]["ner"],
                                      labels_or_path=paths["labels"]["target_domain"]["ner"])]

    # 4. pick a schedule of the selected objectives
    schedule = StridedSchedule(objectives, training_arguments)

    # 5. train using Adapter
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 6. save the trained lang_module (with all heads)
    adapter.save_model("entity_detector_model")


def test_adaptation_translation():
    lang_module = LangModule(test_base_models["translation"], head_types=[Head.LANGUAGE_MODEL])

    # 2. pick objectives - we use BART's objective for adaptation and mBART's seq2seq objective for fine-tuning
    objectives = [DenoisingObjective(lang_module, batch_size=1, texts_or_path=paths["texts"]["target_domain"]["unsup"]),
                  DecoderSequence2Sequence(lang_module, batch_size=1,
                                           texts_or_path=paths["texts"]["target_domain"]["translation"],
                                           labels_or_path=paths["labels"]["target_domain"]["translation"],
                                           source_lang_id="en", target_lang_id="cs")]
    # 3. pick a schedule of the selected objectives
    # this one will shuffle the batches of both objectives
    schedule = StridedSchedule(objectives, training_arguments)

    # 4. train using Adapter
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 5. save the trained (multi-headed) lang_module
    adapter.save_model("translator_model")

    # 6. reload and use it like any other Hugging Face model
    # TODO: some reloads need persisted config
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("translator_model")

# test_adaptation_translation()
