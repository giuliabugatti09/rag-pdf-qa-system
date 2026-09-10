"""
Dicionário de traduções da interface (i18n) e templates de prompt
por idioma, permitindo alternar entre Português e Inglês sem
depender de tradução automática do navegador.
"""

TRANSLATIONS = {
    "pt": {
        # Adicionar dentro de TRANSLATIONS["pt"]
        "status_online": "🟢 API online",
        "status_offline": "🔴 API offline",
        "stats_label": "📚 Base de conhecimento",
        "stats_documents": "{count} documento(s)",
        "stats_chunks": "{count} trechos indexados",
        "examples_label": "💡 Experimente perguntar:",
        "page_title": "RAG — Q&A sobre Documentos",
        "subtitle": "Faça perguntas sobre os documentos indexados na base de conhecimento.",
        "upload_expander": "📤 Adicionar novo documento PDF",
        "upload_button_label": "Escolha um arquivo PDF",
        "upload_process_button": "Processar e adicionar à base",
        "upload_processing": "Processando documento...",
        "upload_success": "'{filename}' adicionado: {pages} páginas, {chunks} chunks.",
        "upload_error": "Erro ao processar o upload: {error}",
        "question_label": "Digite sua pergunta:",
        "question_placeholder": "Ex: Qual foi o lucro líquido em 2023?",
        "ask_button": "Perguntar",
        "searching": "Buscando resposta...",
        "connection_error": "Não foi possível conectar à API. Verifique se o servidor FastAPI está rodando.",
        "request_error": "Erro ao consultar a API: {error}",
        "history_title": "Histórico de perguntas",
        "sources_label": "📎 Fontes",
        "page_label": "pág.",
        "clear_history": "🗑️ Limpar histórico",
        "sidebar_settings": "⚙️ Configurações",
        "sidebar_language": "Idioma / Language",
        "empty_question_warning": "Digite uma pergunta antes de continuar.",
    },
    "en": {
        # Adicionar dentro de TRANSLATIONS["en"]
        "status_online": "🟢 API online",
        "status_offline": "🔴 API offline",
        "stats_label": "📚 Knowledge base",
        "stats_documents": "{count} document(s)",
        "stats_chunks": "{count} indexed chunks",
        "examples_label": "💡 Try asking:",
        "page_title": "RAG — Document Q&A",
        "subtitle": "Ask questions about the documents indexed in the knowledge base.",
        "upload_expander": "📤 Add new PDF document",
        "upload_button_label": "Choose a PDF file",
        "upload_process_button": "Process and add to knowledge base",
        "upload_processing": "Processing document...",
        "upload_success": "'{filename}' added: {pages} pages, {chunks} chunks.",
        "upload_error": "Error processing upload: {error}",
        "question_label": "Type your question:",
        "question_placeholder": "E.g.: What was the net income in 2023?",
        "ask_button": "Ask",
        "searching": "Searching for an answer...",
        "connection_error": "Could not connect to the API. Check if the FastAPI server is running.",
        "request_error": "Error querying the API: {error}",
        "history_title": "Question history",
        "sources_label": "📎 Sources",
        "page_label": "page",
        "clear_history": "🗑️ Clear history",
        "sidebar_settings": "⚙️ Settings",
        "sidebar_language": "Idioma / Language",
        "empty_question_warning": "Please type a question before continuing.",
    },
}


# Instrução de sistema por idioma, para que a LLM responda no idioma
# selecionado na interface, não sempre em português.
PROMPT_INSTRUCTIONS = {
    "pt": {
        "system_role": "Você é um assistente especializado em responder perguntas com base em documentos técnicos e financeiros.",
        "restriction": "Use EXCLUSIVAMENTE as informações do contexto abaixo para responder à pergunta. Não utilize conhecimento prévio ou externo ao contexto fornecido.",
        "fallback": 'Se a informação necessária não estiver presente no contexto, responda claramente: "Não encontrei essa informação nos documentos fornecidos." Não tente adivinhar ou complementar com suposições. Nesse caso, NÃO cite nenhuma página.',
        "citation": "Ao responder, cite a fonte de cada informação relevante, no formato (nome_do_arquivo.pdf, página X).",
        "language_instruction": "Responda sempre em português, independentemente do idioma dos documentos-fonte.",
    },
    "en": {
        "system_role": "You are an assistant specialized in answering questions based on technical and financial documents.",
        "restriction": "Use ONLY the information from the context below to answer the question. Do not use prior or external knowledge.",
        "fallback": 'If the necessary information is not present in the context, respond clearly: "I could not find this information in the provided documents." Do not guess or supplement with assumptions. In that case, do NOT cite any page.',
        "citation": "When answering, cite the source of each relevant piece of information, in the format (filename.pdf, page X).",
        "language_instruction": "Always answer in English, regardless of the source documents' language.",
    },
}


def get_text(key: str, lang: str = "pt", **kwargs) -> str:
    """
    Recupera um texto traduzido pela chave, no idioma especificado,
    aplicando formatação de variáveis quando necessário.

    Args:
        key: chave do texto no dicionário TRANSLATIONS.
        lang: código do idioma ('pt' ou 'en').
        **kwargs: variáveis para formatação (ex: filename, pages).

    Returns:
        O texto traduzido e formatado.
    """
    texto = TRANSLATIONS.get(lang, TRANSLATIONS["pt"]).get(key, key)
    return texto.format(**kwargs) if kwargs else texto

  
