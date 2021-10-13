from domain_adaptation.lang_module import LangModule
from domain_adaptation.utils import Head


def test_lang_model():
    lang_model = LangModule("bert-base-multilingual-cased",
                            head_types=[Head.LANGUAGE_MODEL, Head.TOKEN_CLASSIFICATION])
    return
