"""
Trainingsskript für das durchgängige Kursprojekt.

Trainiert einen einfachen Klassifikator auf dem Iris-Datensatz (in scikit-learn
eingebaut, kein Download nötig) und speichert das Modell als model.pkl.

Dieses Skript wird ab Tag 3 in vielen weiteren Kurstagen wiederverwendet:
- Tag 4: in eine FastAPI-Inference-API eingebettet
- Tag 6: um MLflow-Tracking erweitert
- Tag 9: Daten/Modell werden mit DVC versioniert
- Tag 12/13: als Schritt in eine orchestrierte Pipeline eingebunden
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("Lade Daten...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Trainiere Modell...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy auf Testdaten: {accuracy:.3f}")

    joblib.dump(model, "model.pkl")
    print("Modell gespeichert unter model.pkl")


if __name__ == "__main__":
    main()
