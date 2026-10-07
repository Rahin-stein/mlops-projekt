"""
Promotion-Skript für Tag 13: setzt den @champion-Alias auf die neueste
Modellversion um - läuft nur, wenn das Quality Gate vorher bestanden wurde.
"""

from mlflow.tracking import MlflowClient

MODEL_NAME = "iris-classifier"


def main():
    client = MlflowClient()

    # TODO 1: Holt alle Versionen des Modells.
    # Tipp: client.search_model_versions(f"name='{MODEL_NAME}'")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    # TODO 2: Findet die Version mit der höchsten Versionsnummer.
    # Tipp: max(versions, key=lambda v: int(v.version))
    latest = max(versions, key=lambda v: int(v.version))

    # TODO 3: Setzt den Alias "champion" auf diese Version.
    # Tipp: client.set_registered_model_alias(MODEL_NAME, "champion", ...)
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)

    print(f"@champion auf Version {latest.version} von {MODEL_NAME} gesetzt.")


if __name__ == "__main__":
    main()
