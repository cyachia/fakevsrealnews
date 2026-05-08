import re
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR

LABEL_MAP = {"FAKE": 0, "REAL": 1, "Fake": 0, "Real": 1, 0: 0, 1: 1}


def find_data_file(basename: str) -> Path:
    direct_file = DATA_DIR / basename
    if direct_file.exists():
        return direct_file

    zip_file = DATA_DIR / f"{basename}.zip"
    if zip_file.exists():
        return zip_file

    raise FileNotFoundError(
        f"Keine Datei '{basename}' oder '{basename}.zip' im Ordner '{DATA_DIR}' gefunden."
    )


def load_csv(name: str) -> pd.DataFrame:
    source = find_data_file(name)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            inner_name = name
            if inner_name not in archive.namelist():
                raise FileNotFoundError(
                    f"Die ZIP-Datei '{source.name}' enthält nicht die erwartete Datei '{inner_name}'."
                )
            with archive.open(inner_name) as fp:
                return pd.read_csv(fp)

    return pd.read_csv(source)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_evaluation_data() -> pd.DataFrame:
    test_path = MODEL_DIR / "fake_news_test.csv"
    if test_path.exists():
        print(f"Verwende gespeicherten Holdout-Testdatensatz '{test_path}'.")
        df = pd.read_csv(test_path)
    else:
        print("Gespeicherter Holdout-Testdatensatz nicht gefunden. Verwende 'fake_or_real_news.csv'.")
        df = load_csv("fake_or_real_news.csv")

    if "label" not in df.columns:
        raise ValueError("Die Evaluierungsdaten enthalten keine 'label'-Spalte.")

    df = df[["title", "text", "label"]]
    df["label"] = df["label"].map(LABEL_MAP)

    if df["label"].isnull().any():
        raise ValueError("Es gibt ungültige Label in den Daten. Nur 'FAKE' und 'REAL' sind erlaubt.")

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["clean_content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    df = df.drop_duplicates(subset=["title", "text"], keep="first")
    return df


def load_artifacts():
    model_path = MODEL_DIR / "fake_news_model.joblib"
    vectorizer_path = MODEL_DIR / "fake_news_model_vectorizer.joblib"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Modell oder Vektorisierer wurde nicht gefunden. Bitte zuerst 'train_model.py' ausführen."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def evaluate_model(model, vectorizer, df: pd.DataFrame) -> None:
    X = vectorizer.transform(df["clean_content"])
    y_true = df["label"]
    y_pred = model.predict(X)

    print("Auswertung für fake_or_real_news.csv:")
    print(y_true.value_counts().rename({0: "Fake", 1: "Real"}))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    model, vectorizer = load_artifacts()
    df = load_evaluation_data()
    evaluate_model(model, vectorizer, df)
