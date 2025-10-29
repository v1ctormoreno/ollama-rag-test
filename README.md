> Generado TODO con ChatGPT

# RAG self‑hosted (CPU) con UI


### Qué incluye
- **Ollama** (LLM local) → por defecto `llama3.1:8b-instruct-q4_0` (puedes cambiar la variable `LLM_MODEL`).
- **Qdrant** (vector DB) → almacena embeddings y metadatos de tus documentos.
- **Backend FastAPI** → endpoints `/ingest` y `/chat`.
- **Streamlit UI** → subir ficheros y chatear.


### Requisitos
- Docker y Docker Compose.
- Algo de paciencia en CPU: los modelos quantizados funcionan, pero no esperes latencia de GPU.


### Puesta en marcha
1. Clona/copias estos archivos en una carpeta vacía.
2. Arranca servicios:
```bash
docker compose up -d --build
```
3. En otra terminal, **descarga el modelo en Ollama**:
```bash
# Ejemplos (elige uno)
curl http://localhost:11434/api/pull -d '{"name":"llama3.1:8b-instruct-q4_0"}'
# o un modelo más pequeño si tu CPU sufre
# curl http://localhost:11434/api/pull -d '{"name":"qwen2.5:7b-instruct-q4_0"}'
```
4. Abre la UI: http://localhost:8501
5. Sube PDFs, DOCX o XLSX y lanza preguntas.


### Consejos
- **Excel grandes**: la UI convierte cada hoja a CSV para trocear mejor. Si tu Excel es enorme, considera limpiar columnas irrelevantes antes.
- **Control de alucinaciones**: el `system prompt` obliga a ceñirse al contexto; si falta info, lo dirá.
- **Citas**: en la respuesta verás fuentes con `filename` y un preview. Puedes extender el payload para guardar `sheet`, `page`, etc.
- **Modelos**: en CPU, usa quantizados `q4_0`. Cambia `LLM_MODEL` en el backend (variable de entorno) y vuelve a levantar el servicio.


### Variables útiles
- `LLM_MODEL`: nombre del modelo de Ollama (por defecto `llama3.1:8b-instruct-q4_0`).
- `EMB_MODEL`: embeddings HF (por defecto `sentence-transformers/all-MiniLM-L6-v2`).
- `COLLECTION_NAME`: nombre de la colección en Qdrant.


### Roadmap
- Re-ranking (CrossEncoder) para mejorar precisión.
- Búsqueda híbrida (BM25 + vectorial).
- Autenticación en UI.
- Conversaciones con memoria por usuario.
