"""
Prefect-Flow für Tag 12: verbindet Validierung (Tag 9), Training und
Tracking (Tag 6/8) zu einem zusammenhängenden, nachvollziehbaren Flow.

Füllt die TODOs aus, dann ausführen mit:
    uv run python pipeline.py
"""

import subprocess

from prefect import flow, task


@task  # definiert einen einzelnen Schritt in der Pipeline.
def validate_data():
    print("Validiere Daten...")
    subprocess.run(["python", "validate_data.py"], check=True)


@task
def train_model():
    print("Trainiere Modell...")
    subprocess.run(["python", "train.py"], check=True)


# TODO 2: Baut die Tasks zu einem Flow zusammen - Reihenfolge:
# validate_data, train_model, instabiler_schritt.
# Zunächst (Schritt 4) füllt ihr nur die ersten beiden Zeilen aus; die dritte
# Zeile bleibt vorerst "..." und wird erst in Schritt 5 durch
# instabiler_schritt() ersetzt.
@flow  # macht aus den Tasks eine Pipeline.
def pipeline():
    validate_data()
    train_model()


if __name__ == "__main__":
    pipeline()
