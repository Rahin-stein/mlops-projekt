"""
Inference-API für das durchgängige Kursprojekt (Tag 4).

Füllt die mit TODO markierten Stellen aus. Das Modell (model.pkl) muss im
gleichen Ordner liegen - kopiert es aus eurem Tag-3-Projekt hierher, oder
führt train.py erneut aus, falls ihr es nicht mehr habt.
"""

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris-Klassifikator-API")

# TODO 1: Ladet das trainierte Modell aus model.pkl.
# Wichtig: das passiert hier auf Modul-Ebene, also einmal beim Start der
# Anwendung - nicht innerhalb der Endpoint-Funktion (sonst ladet ihr bei
# jedem Request neu, das wäre viel zu langsam).
model = joblib.load("model.pkl")


# TODO 2: Das Pydantic-Modell unten ist schon fertig - schaut es euch an,
# es beschreibt die vier Merkmale einer Iris-Blüte, die die API erwartet.
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    # TODO 3: Gebt ein einfaches Status-Dict zurück, z. B. {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    # TODO 4: Wandelt die vier Merkmale in ein 2D-numpy-Array um
    # (das Modell erwartet die Form [[wert1, wert2, wert3, wert4]])
    # Tipp: features.sepal_length etc. greifen auf die einzelnen Werte zu
    data = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]
    )

    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
