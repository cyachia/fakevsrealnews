import re
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_CONFIG = {
    "english": {
        "model": "fake_news_model.joblib",
        "vectorizer": "fake_news_model_vectorizer.joblib",
        "label": "Englisches Fake/Real-Modell",
        "description": "Gib hier einen Nachrichtentext mit/oder Überschrift ein. Die App sagt dir, ob der Text eher Fake oder Real ist.",
    },

}


@st.cache_resource
def load_artifacts(model_key: str):
    config = MODEL_CONFIG[model_key]
    model_path = MODEL_DIR / config["model"]
    vectorizer_path = MODEL_DIR / config["vectorizer"]

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Modell oder Vektorisierer wurde nicht gefunden. Bitte zuerst 'train_model.py' ausführen."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(text: str, model, vectorizer) -> tuple[str, float]:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "Real" if pred == 1 else "Fake"

    confidence = float(max(prob) * 100)
    return label, confidence


def main() -> None:
    model, vectorizer = load_artifacts("english")

    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.title("Fake News Detector")
    st.write(
        "Gib hier einen Nachrichtentext oder eine Überschrift ein. Die App sagt dir, ob der Text eher `Fake` oder `Real` ist."
    )

    st.info(
        "Dieses Modell wurde mit gelabelten Fake- und Real-News trainiert und verwendet Logistic Regression für die Klassifikation."
    )

    user_input = st.text_area(
        "Nachricht oder Überschrift",
        height=180,
        placeholder="Zum Beispiel: Regierung plant neue Maßnahmen gegen Klimawandel",
    )

    if st.button("Vorhersage starten"):
        if not user_input.strip():
            st.warning("Bitte zuerst einen Text eingeben.")
            return

        label, confidence = predict_news(user_input, model, vectorizer)
        if label == "Fake":
            st.error(f"Ergebnis: {label} ({confidence:.1f} %)")
        else:
            st.success(f"Ergebnis: {label} ({confidence:.1f} %)")

        st.markdown(
            "---\n"
            "**Hinweis:** Die Vorhersage ist ein statistisches Ergebnis. Bei Echt-Daten kann es Fehler geben."
        )

    st.markdown("---")
    st.write("**So funktioniert die App:**")
    st.write(
        "1. Der Text wird bereinigt: Sonderzeichen und Links werden entfernt."
        " 2. Dann wird der Text in Zahlen umgewandelt (TF-IDF)."
        " 3. Ein trainiertes Logistik-Regression-Modell entscheidet, ob der Text Fake oder Real ist."
    )


if __name__ == "__main__":
    main()