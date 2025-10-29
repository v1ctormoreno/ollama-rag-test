import os, requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Self‑Hosted (CPU)", layout="wide")
st.title("🔎 RAG Self‑Hosted (CPU) – PDFs / Excel / Word")

with st.sidebar:
    st.header("📄 Ingesta de documentos")
    up_file = st.file_uploader("Sube PDF/DOCX/XLSX/TXT", type=["pdf", "docx", "doc", "xlsx", "xls", "txt"], accept_multiple_files=False)
    if up_file and st.button("Ingestar al índice"):
        files = {"file": (up_file.name, up_file.getvalue())}
        r = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=300)
        if r.ok:
            st.success(f"Ingestados {r.json().get('chunks', 0)} trozos de {up_file.name}")
        else:
            st.error(r.text)

st.write("---")

q = st.text_input("Escribe tu pregunta")
col1, col2 = st.columns([1,1])

if st.button("🔍 Consultar") and q.strip():
    with st.spinner("Buscando contexto y generando respuesta..."):
        r = requests.post(f"{BACKEND_URL}/chat", json={"q": q, "top_k": 4}, timeout=180)
    if r.ok:
        data = r.json()
        st.subheader("Respuesta")
        st.write(data.get("answer", ""))
        st.subheader("Fuentes")
        for s in data.get("sources", []):
            with st.expander(f"{s.get('filename','(desconocido)')} – score {s.get('score'):.3f}"):
                st.code(s.get("preview", ""))
    else:
        st.error(r.text)
