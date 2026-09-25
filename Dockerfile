# Dockerfile – Tag 4
# Die Basis kennt ihr schon aus Tag 3 (vorausgefüllt). Neu sind EXPOSE und
# der CMD-Befehl für einen laufenden Server statt eines einmaligen Skripts.

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache-dir -r requirements.txt
COPY main.py .
COPY model.pkl .

# TODO 1: Dokumentiert, auf welchem Port die Anwendung im Container lauscht
# (nur Dokumentation, öffnet den Port noch nicht nach außen - das macht
# gleich der `docker run -p`-Befehl)
EXPOSE 8000

# TODO 2: Startet den Uvicorn-Server. Syntax:
# uvicorn <dateiname_ohne_.py>:<fastapi-variable> --host 0.0.0.0 --port <port>
# --host 0.0.0.0 ist wichtig, sonst ist die API von außerhalb des
# Containers nicht erreichbar!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
