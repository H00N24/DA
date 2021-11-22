from domain_adaptation.lang_module import LangModule


def test_lang_model():
    lang_model = LangModule("bert-base-multilingual-cased")
    assert lang_model
