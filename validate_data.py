"""
Datenvalidierung mit Pandera für data/iris.csv.

Füllt die TODOs aus, dann ausführen mit:
    uv run python validate_data.py
"""

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

# TODO 1: Definiert ein Schema für die vier numerischen Merkmalsspalten.
# Alle vier sind Fließkommazahlen und sollten sinnvollerweise zwischen
# 0 und 10 liegen (echte Blütenmaße in cm). "target" ist eine Ganzzahl
# zwischen 0 und 2 (drei Iris-Arten).
schema = pa.DataFrameSchema(
    {
        "sepal_length": Column(float, Check.in_range(0, 10)),
        "sepal_width": Column(float, Check.in_range(0, 10)),
        "petal_length": Column(float, Check.in_range(0, 10)),
        "petal_width": Column(float, Check.in_range(0, 10)),
        "target": Column(int, Check.isin([0, 1, 2])),
    }
)
df = pd.read_csv("data/iris.csv")

try:
    # TODO 2: Validiert df gegen das Schema (schema.validate(...))
    validated = schema.validate(df)
    print(
        f"Validierung erfolgreich: {len(validated)} Zeilen geprüft, keine Auffälligkeiten."
    )
except pa.errors.SchemaError as e:
    print("Validierung fehlgeschlagen:")
    print(e)

# schritt 7 :
# Hier prüfen wie die Daten mit Pandera.
# Ich habe Regeln für die Daten definiert.
# Pandera prüft, ob die Werte korrekt sind.                                       uv run python validate_data.py
# Hier verstehen wir, dass alle 150 Zeilen in Ordnung sind.

# schrit 8 :
# Hier habe ich absichtlich einen falschen Wert, 999, eingefügt.                  sed -i '2s/^[^,]*/999/' data/iris.csv
# Pandera erkennt den Fehler, weil der Wert nicht zwischen 0 und 10 liegt.        uv run python validate_data.py

# Jetzt stellen wir die richtige Daten wieder zurück.                             uv run dvc checkout --force
#                                                                                 uv run python validate_data.py
# schritt 9 :
# Hier löche ich die lokale Datei                                                 rm -rf .dvc/cache data/iris.csv
#                                                                                 uv run dvc status
#  ich herunterlade die Daten wieder aus MinIO                                    uv run dvc pull
