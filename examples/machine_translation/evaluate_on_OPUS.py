"""
Example script Machine Translation adaptation
We train a NMT model on Wikipedia parallel corpus while adapting it to OpenSubtitles domain.

We perform the following steps:
1. Load datasets: once available, this can be rewritten for HF Datasets library
2. Perform a combined adaptation on both parallel data and monolingual, OpenSubtitles domain: Strided schedule.
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
    lm_model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, config=config)

    assert hasattr(lm_model, "generate"), "For translation, we need a model that implements its own generate()."

    translations = []

    for src_text in tqdm(adapted_test_dataset.source):
        inputs = tokenizer(src_text, truncation=True, return_tensors="pt")
        outputs = lm_model.generate(**inputs)
        translations.extend(tokenizer.batch_decode(outputs))
        if len(translations) % 10 == 0:
            print("BLEU: %s" % BLEU.evaluate_str(translations, adapted_test_dataset.target))

    bleu = BLEU.evaluate_str(translations, adapted_test_dataset.target)

    print("Test BLEU on %s (%s->%s): %s" % (args.opus_dataset, args.src_lang, args.tgt_lang, bleu))
    print("Done")
