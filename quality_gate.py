"""
Quality Gate für Tag 13: liest die Metrik des neuesten MLflow-Runs aus
und bricht mit Fehlercode ab, wenn sie unter dem Schwellenwert liegt.
"""

import sys

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
THRESHOLD = 0.9


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-klassifikator")

    # TODO 1: Holt den neuesten Run des Experiments.
    # Tipp: client.search_runs(experiment.experiment_id,
    #       order_by=["start_time DESC"], max_results=1)
    runs = client.search_runs(
        experiment.experiment_id, order_by=["start_time DESC"], max_results=1
    )
    latest_run = runs[0]

    # TODO 2: Liest die "accuracy"-Metrik aus latest_run.data.metrics
    accuracy = latest_run.data.metrics["accuracy"]

    print(f"Neueste Accuracy: {accuracy:.3f} (Schwellenwert: {THRESHOLD})")

    if accuracy < THRESHOLD:
        print("Quality Gate NICHT bestanden.")
        sys.exit(1)

    print("Quality Gate bestanden.")


if __name__ == "__main__":
    main()
