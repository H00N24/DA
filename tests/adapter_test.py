from domain_adaptation.adapter import Adapter
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.MLM import MaskedLanguageModeling
from domain_adaptation.objectives.classification import TokenClassification
from domain_adaptation.schedules import SequentialSchedule
from domain_adaptation.utils import AdaptationArguments, StoppingStrategy
from utils import paths, test_base_models


training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=2,
                                         log_level="critical",
                                         logging_steps=1,
                                         num_train_epochs=3)


def run_adaptation(adapter: Adapter, trained_model_output_dir: str = "adaptation_output_dir/finished"):
    adapter.train()
    adapter.save_model(trained_model_output_dir)


def test_ner_adaptation():
    lang_module = LangModule(test_base_models["token_classification"])
    objectives = [
            MaskedLanguageModeling(lang_module,
                                   texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                   batch_size=1),
            TokenClassification(lang_module,
                                texts_or_path=paths["texts"]["target_domain"]["ner"],
                                labels_or_path=paths["labels"]["target_domain"]["ner"],
                                batch_size=1)
    ]

    schedule = SequentialSchedule(objectives, training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)

    run_adaptation(adapter)
