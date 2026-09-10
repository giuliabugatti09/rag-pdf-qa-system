"""
Interface visual do sistema RAG, construída com Streamlit.

Consome a API FastAPI (não importa a lógica RAG diretamente), validando
que a arquitetura está corretamente desacoplada entre backend e frontend.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄", layout="centered")

st.title("📄 RAG — Q&A sobre Documentos PDF")
st.caption("Faça perguntas sobre os documentos indexados na base de conhecimento.")

# Inicializa o histórico de conversa na sessão, se ainda não existir
if "historico" not in st.session_state:
    st.session_state.historico = []

# --- Seção de upload de novos documentos ---
with st.expander("📤 Adicionar novo documento PDF"):
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Processar e adicionar à base"):
        with st.spinner("Processando documento..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                response.raise_for_status()
                resultado = response.json()
                st.success(
                    f"'{resultado['filename']}' adicionado: "
                    f"{resultado['pages_processed']} páginas, "
                    f"{resultado['chunks_added']} chunks."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao processar o upload: {e}")

st.divider()

# --- Seção de pergunta ---
pergunta = st.text_input("Digite sua pergunta:", placeholder="Ex: Qual foi o lucro líquido em 2023?")

if st.button("Perguntar", type="primary") and pergunta:
    with st.spinner("Buscando resposta..."):
        try:
            response = requests.post(f"{API_URL}/query", json={"question": pergunta})
            response.raise_for_status()
            resultado = response.json()

            # Adiciona ao histórico da sessão
            st.session_state.historico.insert(0, resultado)

        except requests.exceptions.ConnectionError:
            st.error(
                "Não foi possível conectar à API. Verifique se o servidor "
                "FastAPI está rodando (uvicorn src.api.main:app)."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao consultar a API: {e}")

# --- Exibição do histórico ---
if st.session_state.historico:
    st.subheader("Histórico de perguntas")

    for item in st.session_state.historico:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

            if item.get("sources"):
                fontes_texto = ", ".join(
                    f"{s['arquivo']} (pág. {s['pagina']})" for s in item["sources"]
                )
                st.caption(f" Fontes: {fontes_texto}")