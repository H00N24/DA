from domain_adaptation.evaluators.generative import BLEU
from domain_adaptation.lang_module import LangModule
from domain_adaptation.objectives.objective_base import Objective
from domain_adaptation.objectives.seq2seq import DecoderSequence2Sequence
from utils import paths, test_base_models


def assert_evaluator_logs(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(**dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"], split)

    log = objective.per_objective_log(split, aggregation_steps=1)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert all(str(objective) in k for k in log.keys())


def test_train_bleu_with_seq2seq():
    lang_module = LangModule(test_base_models["translation"])

    train_bleu = BLEU()
    val_bleu = BLEU(decides_convergence=True)
    objective = DecoderSequence2Sequence(lang_module,
                                         texts_or_path=paths["texts"]["target_domain"]["translation"],
                                         labels_or_path=paths["labels"]["target_domain"]["translation"],
                                         batch_size=1,
                                         source_lang_id="en",
                                         target_lang_id="cs",
                                         train_evaluators=[train_bleu],
                                         val_evaluators=[val_bleu])

    assert_evaluator_logs(lang_module, objective)

