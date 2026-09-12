"""
Interface visual do sistema RAG, com suporte bilíngue (PT/EN),
status da API, estatísticas da base e perguntas de exemplo.
"""

import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
from src.ui.i18n import get_text

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄", layout="centered")

# --- CSS customizado: cards de resposta com borda suave ---
st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 12px;
        padding: 4px;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(250, 250, 250, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Estado inicial da sessão ---
if "lang" not in st.session_state:
    st.session_state.lang = "pt"
if "historico" not in st.session_state:
    st.session_state.historico = []
if "pergunta_input" not in st.session_state:
    st.session_state.pergunta_input = ""

lang = st.session_state.lang


def t(key: str, **kwargs) -> str:
    return get_text(key, lang=lang, **kwargs)


def get_api_status() -> bool:
    """Verifica se a API está respondendo, com timeout curto."""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_stats() -> dict | None:
    """Busca estatísticas da base de conhecimento, se a API estiver no ar."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# --- Sidebar ---
with st.sidebar:
    st.header(t("sidebar_settings"))

    idioma_selecionado = st.radio(
        t("sidebar_language"),
        options=["pt", "en"],
        format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English",
        index=0 if lang == "pt" else 1,
    )

    if idioma_selecionado != st.session_state.lang:
        st.session_state.lang = idioma_selecionado
        st.rerun()

    st.divider()

    # Status da API
    api_online = get_api_status()
    st.markdown(t("status_online") if api_online else t("status_offline"))

    # Estatísticas da base
    if api_online:
        stats = get_stats()
        if stats:
            st.caption(t("stats_label"))
            st.write(t("stats_documents", count=stats["total_documentos"]))
            st.write(t("stats_chunks", count=stats["total_chunks"]))

    st.divider()

    if st.session_state.historico and st.button(t("clear_history")):
        st.session_state.historico = []
        st.rerun()


# --- Cabeçalho ---
st.title(f"📄 {t('page_title')}")
st.caption(t("subtitle"))

# --- Upload de novos documentos ---
with st.expander(t("upload_expander")):
    uploaded_file = st.file_uploader(t("upload_button_label"), type=["pdf"])

    if uploaded_file is not None and st.button(t("upload_process_button")):
        with st.spinner(t("upload_processing")):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                response.raise_for_status()
                resultado = response.json()
                st.success(
                    t(
                        "upload_success",
                        filename=resultado["filename"],
                        pages=resultado["pages_processed"],
                        chunks=resultado["chunks_added"],
                    )
                )
            except requests.exceptions.RequestException as e:
                st.error(t("upload_error", error=str(e)))

st.divider()

# --- Perguntas de exemplo ---
EXEMPLOS = {
    "pt": [
        "Qual foi o lucro líquido em 2023?",
        "Qual foi o EBITDA ajustado da Vale?",
        "O que é Retrieval-Augmented Generation?",
    ],
    "en": [
        "What was the net income in 2023?",
        "What was Vale's adjusted EBITDA?",
        "What is Retrieval-Augmented Generation?",
    ],
}

st.caption(t("examples_label"))
cols = st.columns(len(EXEMPLOS[lang]))
for col, exemplo in zip(cols, EXEMPLOS[lang]):
    if col.button(exemplo, use_container_width=True):
        st.session_state["pergunta_widget"] = exemplo
        st.rerun()

# --- Formulário de pergunta ---
pergunta = st.text_input(
    t("question_label"),
    placeholder=t("question_placeholder"),
    key="pergunta_widget",
)

if st.button(t("ask_button"), type="primary"):
    if not pergunta.strip():
        st.warning(t("empty_question_warning"))
    else:
        with st.spinner(t("searching")):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": pergunta, "language": lang},
                )
                response.raise_for_status()
                resultado = response.json()
                st.session_state.historico.insert(0, resultado)
                st.session_state.pergunta_input = ""
            except requests.exceptions.ConnectionError:
                st.error(t("connection_error"))
            except requests.exceptions.RequestException as e:
                st.error(t("request_error", error=str(e)))

# --- Histórico de conversa ---
if st.session_state.historico:
    st.subheader(t("history_title"))

    for item in st.session_state.historico:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

            if item.get("sources"):
                fontes_texto = ", ".join(
                    f"{s['arquivo']} ({t('page_label')} {s['pagina']})"
                    for s in item["sources"]
                )
                st.caption(f"{t('sources_label')}: {fontes_texto}")