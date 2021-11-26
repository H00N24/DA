"""
TODO: comment
"""
import argparse

from tqdm import tqdm
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer

from domain_adaptation.evaluators.generative import BLEU
from examples.opus import OPUSDataset, OPUS_RESOURCES_URLS

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, help='Path of the fine-tuned checkpoint.',
                           required=True)
    argparser.add_argument('--config_file_path', type=str, help='Path of the config of the used head.',
                           required=True)
    argparser.add_argument('--opus_dataset', type=str, required=True,
                           help='One of the recognized OPUS datasets: %s' % OPUS_RESOURCES_URLS)
    argparser.add_argument('--src_lang', type=str, required=True, help='Source language of the OPUS data set.')
    argparser.add_argument('--tgt_lang', type=str, required=True, help='Target language of the OPUS data set.')
    argparser.add_argument('--data_dir', type=str, required=True, help='Cache directory to store the data.')
    argparser.add_argument('--firstn', type=int, default=None, help='If given, subsets data set to first-n samples.')
    argparser.add_argument('--device', type=str, default="cpu", help='Device for inference. Defaults to CPU.')
    args = argparser.parse_args()

    adapted_test_dataset = OPUSDataset(args.opus_dataset, src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                                       split="test", data_dir=args.data_dir, firstn=args.firstn)

    config = AutoConfig.from_pretrained(args.config_file_path)
    lm_model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, config=config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, config=config)

    assert hasattr(lm_model, "generate"), "For translation, we need a model that implements its own generate()."

    translations = []
    references = []

    for src_text, ref_text in tqdm(zip(adapted_test_dataset.source, adapted_test_dataset.target),
                                   total=len(adapted_test_dataset.source)):
        inputs = tokenizer(src_text, truncation=True, return_tensors="pt").to(args.device)
        outputs = lm_model.generate(**inputs)
        translations.extend(tokenizer.batch_decode(outputs))
        references.append(ref_text)
        if len(translations) % 10 == 0:
            print("BLEU: %s" % BLEU.evaluate_str(translations, references))

    bleu = BLEU.evaluate_str(translations, adapted_test_dataset.target)

    print("Test BLEU on %s (%s->%s): %s" % (args.opus_dataset, args.src_lang, args.tgt_lang, bleu))
    print("Done")
