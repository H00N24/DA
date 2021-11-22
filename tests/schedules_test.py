from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
from domain_adaptation.schedules import SequentialSchedule, TrainingSchedule, StridedSchedule
from domain_adaptation.utils import AdaptationArguments, StoppingStrategy
from utils import test_base_models

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"

sup_translation_texts_src = "mock_data/seq2seq_sources.txt"
sup_translation_texts_tgt = "mock_data/seq2seq_targets.txt"

args = AdaptationArguments(output_dir="adaptation_output_dir",
                           stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                           do_train=True,
                           do_eval=True,
                           gradient_accumulation_steps=2,
                           log_level="critical",
                           logging_steps=1,
                           num_train_epochs=20)


def assert_schedule(lang_module: LangModule, schedule: TrainingSchedule, split: str = "train"):
    dataset_iter = iter(schedule.iterable_dataset(split))
    for batch in dataset_iter:

        logit_outputs = lang_module(**batch)

        loss_combined = schedule.compute_loss(logit_outputs, batch["labels"])
        loss_combined.backward()

        assert True

    assert True


def ner_da_schedule(schedule_type):
    lang_module = LangModule(test_base_models["token_classification"])

    lm_adaptation = MaskedLanguageModeling(lang_module, texts_or_path=unsup_target_domain_texts, batch_size=1)
    token_classification = TokenClassification(lang_module, texts_or_path=sup_target_domain_texts,
                                               labels_or_path=sup_target_domain_labels, batch_size=1)

    assert_schedule(lang_module, schedule_type(objectives=[lm_adaptation, token_classification], args=args))


def test_ner_da_schedule_sequential():
    ner_da_schedule(SequentialSchedule)


def test_ner_da_schedule_strided():
    ner_da_schedule(StridedSchedule)


def test_mt_da_schedule():
    lang_module = LangModule(test_base_models["translation"])
    denoising_adaptation = DenoisingObjective(lang_module, texts_or_path=unsup_target_domain_texts, batch_size=1)
    clm_finetuning = DecoderSequence2Sequence(lang_module, texts_or_path=sup_translation_texts_src,
                                              labels_or_path=sup_translation_texts_tgt, source_lang_id="en",
                                              target_lang_id="cs", batch_size=1)

    assert_schedule(lang_module, SequentialSchedule(objectives=[denoising_adaptation, clm_finetuning], args=args))
