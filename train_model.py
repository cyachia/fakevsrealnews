import re
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


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


LABEL_MAP = {"FAKE": 0, "REAL": 1, "Fake": 0, "Real": 1, 0: 0, 1: 1}


def load_data() -> pd.DataFrame:
    try:
        df = load_csv("fake_or_real_news.csv")
        print("Verwende 'fake_or_real_news.csv' als Hauptdatensatz.")
    except FileNotFoundError:
        print("'fake_or_real_news.csv' nicht gefunden, verwende 'Fake.csv' und 'True.csv'.")
        dfs = []
        try:
            fake = load_csv("Fake.csv")
            fake["label"] = 0
            dfs.append(fake)
        except FileNotFoundError:
            pass

        try:
            true = load_csv("True.csv")
            true["label"] = 1
            dfs.append(true)
        except FileNotFoundError:
            pass

        if not dfs:
            raise FileNotFoundError("Keine Datendateien gefunden.")

        df = pd.concat(dfs, ignore_index=True)

    df = df[["title", "text", "label"]]
    df["label"] = df["label"].map(LABEL_MAP)

    if df["label"].isnull().any():
        raise ValueError("Es gibt ungültige Label in den Daten. Nur 'FAKE' und 'REAL' sind erlaubt.")

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["clean_content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    df = df.drop_duplicates(subset=["title", "text"], keep="first")

    return df


def evaluate_model(model, vectorizer, df: pd.DataFrame, dataset_name: str) -> None:
    X = vectorizer.transform(df["clean_content"])
    y_true = df["label"]
    y_pred = model.predict(X)

    print(f"\nAuswertung für {dataset_name}:")
    print(y_true.value_counts().rename({0: "Fake", 1: "Real"}))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification report:\n", classification_report(y_true, y_pred, digits=4))


def train_model() -> None:
    df = load_data()
    print("Geladene Trainingsdaten:")
    print(df["label"].value_counts().rename({0: "Fake", 1: "Real"}))

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(df_train["clean_content"])
    X_test_vec = vectorizer.transform(df_test["clean_content"])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, df_train["label"])

    print("\nAuswertung auf dem Holdout-Testdatensatz:")
    evaluate_model(model, vectorizer, df_test, "Holdout-Testdatensatz")

    test_path = MODEL_DIR / "fake_news_test.csv"
    df_test[["title", "text", "label"]].to_csv(test_path, index=False)
    print(f"Holdout-Testdatensatz gespeichert in '{test_path}'.")

    # Finales Modell auf dem kompletten Datensatz trainieren
    X_full = vectorizer.fit_transform(df["clean_content"])
    model.fit(X_full, df["label"])

    joblib.dump(model, MODEL_DIR / "fake_news_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "fake_news_model_vectorizer.joblib")
    print(f"Modell und Vektorisierer in '{MODEL_DIR}' gespeichert.")


if __name__ == "__main__":
    train_model()
