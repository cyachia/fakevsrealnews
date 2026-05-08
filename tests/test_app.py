import app


def test_clean_text_removes_links_and_punctuation():
    text = "Dies ist ein Test. Mehr Infos unter https://example.com!"
    cleaned = app.clean_text(text)
    assert "https" not in cleaned
    assert "." not in cleaned
    assert "mehr infos" in cleaned


def test_predict_news_returns_label_and_confidence():
    class DummyModel:
        def predict(self, X):
            return [1]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

    class DummyVectorizer:
        def transform(self, texts):
            return texts

    label, confidence = app.predict_news(
        "Dies ist ein sehr langer Testtext mit mehr als zehn Wörtern.",
        DummyModel(),
        DummyVectorizer(),
    )

    assert label == "Real"
    assert confidence == 80.0
