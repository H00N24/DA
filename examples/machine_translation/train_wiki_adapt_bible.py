"""
Example script Machine Translation adaptation
We train a NMT model on Wikipedia parallel corpus while adapting it to OpenSubtitles domain.

We perform the following steps:
1. Load datasets: once available, this can be rewritten for HF Datasets library
2. Perform a combined adaptation on both parallel data and monolingual, OpenSubtitles domain: Strided schedule.
"""
from domain_adaptation.adapter import Adapter
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.CLM import DecoderSequence2Sequence
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.schedules import StridedSchedule, SequentialSchedule
from domain_adaptation.utils import AdaptationArguments, StoppingStrategy, Head

# 1. Load datasets
from examples.opus import OPUSDataset

tmp_data_dir = "."

train_source = OPUSDataset("wikimedia", split="train", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=10)
train_val_source = OPUSDataset("wikimedia", split="val", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=8)

adapt_source = OPUSDataset("Bible", split="train", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=10)
adapt_val_source = OPUSDataset("Bible", split="val", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=8)


# 2. Perform a combined adaptation on both parallel data and monolingual, OpenSubtitles domain: Strided schedule.
training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGE,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=4,
                                         logging_steps=100,
                                         eval_steps=2000,
                                         num_train_epochs=20,
                                         evaluation_strategy="steps",
                                         dataloader_pin_memory=False)
lang_module = LangModule("Helsinki-NLP/opus-mt-en-cs", head_types=[Head.LANGUAGE_MODEL])
lang_module.reinitialize()

clm_training = DecoderSequence2Sequence(lang_module,
                                        texts_or_path=train_source.source,
                                        labels_or_path=train_source.target,
                                        val_texts_or_path=train_val_source.source,
                                        val_labels_or_path=train_val_source.target,
                                        source_lang_id="en", target_lang_id="cs", batch_size=16)

denoising_adaptation = DenoisingObjective(lang_module,
                                          texts_or_path=adapt_source.source,
                                          val_texts_or_path=adapt_val_source.source,
                                          batch_size=16)

schedule = StridedSchedule(objectives=[clm_training, denoising_adaptation], args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()

adapter.save_model(tmp_data_dir)
print("Adaptation finished. Trained model can be reloaded from %s" % tmp_data_dir)

# notes:
# 1. num_train_epochs is not really transparent -> it will stop after given iteration regardless stopping_strategy
# trainer still does its own iteration, reiniting dataloader per-epoch
# 2. we might want to allow the use of Objective only for evaluationWe train a NMT model on Wikipedia parallel corpus while adapting it to OpenSubtitles domain.