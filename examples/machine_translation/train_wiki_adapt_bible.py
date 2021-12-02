"""
Example script Machine Translation adaptation
We train a NMT model on Wikipedia parallel corpus while adapting it to OpenSubtitles domain.

We perform the following steps:
1. Load datasets: once available, this can be rewritten for HF Datasets library
2. Perform a combined adaptation on both parallel data and monolingual, OpenSubtitles domain: Strided schedule.
"""
import comet_ml  # TODO: resolve this smarter

from domain_adaptation.adapter import Adapter
from domain_adaptation.evaluators.generative import BLEU, ROUGE, BERTScore
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.denoising import DenoisingObjective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
from domain_adaptation.schedules import StridedSchedule, SequentialSchedule
from domain_adaptation.utils import AdaptationArguments, StoppingStrategy

# 1. Load datasets
from examples.opus import OPUSDataset

tmp_data_dir = "adaptation_output_dir"
output_model_dir = "adapted_model"

train_source = OPUSDataset("wikimedia", split="train", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir)
train_val_source = OPUSDataset("wikimedia", split="val", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=300)

adapt_source = OPUSDataset("Bible", split="train", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir)
adapt_val_source = OPUSDataset("Bible", split="val", src_lang="en", tgt_lang="cs", data_dir=tmp_data_dir, firstn=300)

# 2. Perform a combined adaptation on both parallel data and monolingual, OpenSubtitles domain: Strided schedule.
training_arguments = AdaptationArguments(output_dir=output_model_dir,
                                         learning_rate=2e-4,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=10000,
                                         gradient_accumulation_steps=4,
                                         logging_steps=105,
                                         eval_steps=1000,
                                         save_steps=1000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps")
lang_module = LangModule("Helsinki-NLP/opus-mt-en-cs")
lang_module.reinitialize()

train_bleu = BLEU()
val_bleu = BLEU(decides_convergence=True)

metrics_args = {"additional_sep_char": "▁"}

train_metrics = [BLEU(**metrics_args), ROUGE(**metrics_args), BERTScore(**metrics_args)]
val_metrics = [BLEU(**metrics_args), ROUGE(**metrics_args), BERTScore(**metrics_args, decides_convergence=True)]

denoising_adaptation = DenoisingObjective(lang_module,
                                          texts_or_path=adapt_source.source,
                                          val_texts_or_path=adapt_val_source.source,
                                          batch_size=8,
                                          train_evaluators=train_metrics,
                                          val_evaluators=val_metrics)

clm_training = DecoderSequence2Sequence(lang_module,
                                        texts_or_path=train_source.source,
                                        labels_or_path=train_source.target,
                                        val_texts_or_path=train_val_source.source,
                                        val_labels_or_path=train_val_source.target,
                                        source_lang_id="en",
                                        target_lang_id="cs",
                                        batch_size=8,
                                        train_evaluators=train_metrics,
                                        val_evaluators=val_metrics)

schedule = StridedSchedule(objectives=[denoising_adaptation, clm_training], args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()

adapter.save_model(output_model_dir)
print("Adaptation finished. Trained model can be reloaded from path: `%s`" % tmp_data_dir)

# notes:
# 1. num_train_epochs is not really transparent -> it will stop after given iteration regardless stopping_strategy
# trainer still does its own iteration, reiniting dataloader per-epoch
# 2. we might want to allow the use of Objective only for evaluationWe train a NMT model on Wikipedia parallel corpus
# while adapting it to OpenSubtitles domain.
