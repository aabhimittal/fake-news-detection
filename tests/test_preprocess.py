from fakenews.preprocess import clean_text, clean_corpus


def test_lowercases_and_strips_urls_and_handles():
    raw = "BREAKING!! Visit https://spam.example NOW @user #tag"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "@user" not in cleaned
    assert "#tag" not in cleaned
    assert cleaned == cleaned.lower()


def test_collapses_whitespace():
    assert clean_text("a\n\n  b\t c") == "a b c"


def test_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"


def test_stopword_removal_optional():
    assert "the" in clean_text("the cat")
    assert "the" not in clean_text("the cat", remove_stopwords=True)


def test_clean_corpus_returns_list():
    out = clean_corpus(["A", "B"])
    assert out == ["a", "b"]
