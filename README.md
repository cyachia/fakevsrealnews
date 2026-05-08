# Fake vs Real: News Detector

## Projektübersicht

Projekttitel: Fake vs Real News Detector

Auftraggeber und Problem:
Dieses Projekt soll Mahreens Vater dabei behilflich sein, einen Nachrichtentext schnell als potenziell "Fake" oder "Real" einzuschätzen. Das Problem ist, dass im täglichen Umfeld viele Texte, Headlines und Social-Media-Meldungen nicht so einfach auf ihren Wahrheitsgehalt überprüft werden können.

Kurze Beschreibung der Lösung und Vorgehensweise:
Die Anwendung verwendet ein mit Nachrichten trainiertes Machine-Learning-Modell, um eingegebene Texte automatisiert zu klassifizieren. Der User gibt einen Text ein, die App bereinigt ihn, wandelt ihn in TF-IDF-Merkmale um und trifft dann mit logistischer Regression eine Vorhersage.

Öffentlicher Link:
[Noch nicht veröffentlicht. Bitte hier den Live-Link einfügen, sobald die Streamlit-App online ist.]

## Benutzung

1. Installieren Sie die Abhängigkeiten mit:
   ```bash
   pip install -r requirements.txt
   ```
2. Trainieren Sie das Modell (falls noch nicht vorhanden):
   ```bash
   python train_model.py
   ```
3. Starten Sie die Anwendung:
   ```bash
   streamlit run app.py
   ```
4. Text in das Eingabefeld einfügen und auf "Vorhersage starten" drücken.

## ML-Anteil

Das Projekt verwendet folgenden ML-Workflow:
- Datensammlung aus realen Fake-/Real-News-Datensätzen
- Textbereinigung und Vorverarbeitung
- TF-IDF-Vektorisierung
- Klassifikation mit `LogisticRegression`
- Evaluation des Holdout-Testdatensatzes

Die Hauptarbeit liegt in der Auswahl und Vorbereitung von Textdaten, dem Training des Modells und der Integration der Vorhersage in die Web-App.

## Verwendete Tools und Libraries

- Python
- Streamlit
- pandas
- scikit-learn
- joblib
- pytest

## Testhinweise

- Unit-Test für Textbereinigung und Vorhersage-Logik: `pytest -q`
- Zusätzlich sollten folgende manuelle Tests gemacht werden:
  - Eingabe eines normalen Nachrichtentexts überprüfen
  - Eingabe eines kurzen Texts prüfen (Warnung anzeigen)
  - Leeres Eingabefeld prüfen
  - Überprüfen, dass Ergebnis und Konfidenz angezeigt werden

## Grenzen und bekannte Probleme

- Das Modell kann nur die statistische Wahrscheinlichkeiten einschätzen, es kann nicht, zum Beispiel, die im Text verwendeten Wörter deuten.
- Sehr kurze oder unklare Texte führen eher zu unzuverlässigen Ergebnissen. 
- Der Trainingsdatensatz kann Verzerrungen enthalten und ist nicht für alle Themenbereiche repräsentativ.
- Bei stark verändertem Schreibstil oder neuen Ereignissen (Aktuelles Weltgeschehen) kann die Klassifikation schlechter werden.

## Quellen

- Fake News Dataset: `fake_or_real_news.csv.zip`, `Fake.csv.zip`, `True.csv.zip` (Quelle: [Kaggle.](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/discussion?sort=hotness))
- scikit-learn Dokumentation

## KI / Agenten

Dieses Repository wurde mit Unterstützung eines Programmieragenten (Github Copilot) in VS Code bearbeitet. Die Vorgehensweise wurde mithilfe von Chatgpt optimiert. Auch wurde Deepseek für Erklärungen und Änderungen verwendet. 
## Arbeitsaufteilung (Gruppenarbeit)
Dieses Projekt wurde zu zweit bearbeitet. Die Arbeit wurde (grob) so aufgeteilt. 

- Mahreen: Datenverarbeitung, Streamlit-Webapp-Entwicklung Modelltrainingwahl und -training
- Shourya: Modellwahl und -training, Dokumentation, kleinere Änderungen, 

