import sys
import pytest

@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Conhecida limitação do ChromaDB (HNSWLib/mmap) no Windows: "
        "o lock de arquivo pode persistir por tempo indeterminado ao "
        "recriar um índice dentro do mesmo processo, mesmo após "
        "gc.collect() e limpeza de cache do client. A idempotência "
        "em si (build_vectorstore não duplicar dados) é garantida "
        "pela lógica de remoção do índice anterior, validada "
        "manualmente nos Dias 5 e 16 do desenvolvimento."
    ),
)

def test_rebuilding_vectorstore_does_not_duplicate(monkeypatch):
    """
    Chamar build_vectorstore() duas vezes seguidas com os mesmos chunks
    não deve resultar em itens duplicados no banco.
    """
    import src.retrieval.vectorstore as vs_module
    monkeypatch.setattr(vs_module, "PERSIST_DIRECTORY", TEST_PERSIST_DIR)

    _remover_com_retry(TEST_PERSIST_DIR)

    chunks = [
        Document(page_content="Chunk de teste um.", metadata={"source": "teste.pdf", "page": 0}),
        Document(page_content="Chunk de teste dois.", metadata={"source": "teste.pdf", "page": 1}),
    ]

    vectorstore1 = None
    vectorstore2 = None

    try:
        vectorstore1 = build_vectorstore(chunks)
        count_after_first = vectorstore1._collection.count()

        del vectorstore1
        gc.collect()
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        time.sleep(1.0)

        vectorstore2 = build_vectorstore(chunks)
        count_after_second = vectorstore2._collection.count()

        assert count_after_first == len(chunks)
        assert count_after_second == len(chunks), (
            "Reconstruir o vector store duplicou os chunks "
        )
    finally:
        del vectorstore2
        gc.collect()
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        time.sleep(1.0)
        _remover_com_retry(TEST_PERSIST_DIR)