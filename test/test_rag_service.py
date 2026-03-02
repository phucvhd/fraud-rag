import pytest
from unittest.mock import patch, MagicMock
from services.api.rag_service import RAGQueryEngine

@patch("services.api.rag_service.OpenAI")
@patch("services.api.rag_service.SentenceTransformer")
@patch("services.api.rag_service.create_engine")
@patch("services.api.rag_service.config_loader")
def test_rag_query_engine_init(mock_config_loader, mock_create_engine, mock_sentence_transformer, mock_openai):
    mock_config = MagicMock()
    mock_config_loader.load.return_value = mock_config

    engine = RAGQueryEngine()

    mock_create_engine.assert_called_once()
    mock_sentence_transformer.assert_called_once()
    mock_openai.assert_called_once()
    
@patch("services.api.rag_service.OpenAI")
@patch("services.api.rag_service.SentenceTransformer")
@patch("services.api.rag_service.create_engine")
@patch("services.api.rag_service.config_loader")
def test_rag_query_engine_retrieve_context(mock_config_loader, mock_create_engine, mock_sentence_transformer, mock_openai):
    mock_config_loader.load.return_value = MagicMock()
    
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    mock_sentence_transformer.return_value = mock_embedder

    mock_db_engine = MagicMock()
    mock_conn = MagicMock()
    mock_db_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_db_engine
    
    mock_records = [{"amount": 100, "event_timestamp": "2023", "embedding_text": "text"}]
    mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_records

    engine = RAGQueryEngine()
    
    context = engine._retrieve_context("query", 5)
    
    assert context == mock_records
    mock_embedder.encode.assert_called_once_with("query")

@patch("services.api.rag_service.OpenAI")
@patch("services.api.rag_service.SentenceTransformer")
@patch("services.api.rag_service.create_engine")
@patch("services.api.rag_service.config_loader")
def test_rag_query_engine_generate_response(mock_config_loader, mock_create_engine, mock_sentence_transformer, mock_openai):
    mock_config_loader.load.return_value = MagicMock()
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    engine = RAGQueryEngine()
    mock_context = [{"amount": 100, "event_timestamp": "2023", "embedding_text": "text"}]
    answer = engine._generate_response("query", mock_context)

    assert answer == "mocked answer"

@patch("services.api.rag_service.OpenAI")
@patch("services.api.rag_service.SentenceTransformer")
@patch("services.api.rag_service.create_engine")
@patch("services.api.rag_service.config_loader")
def test_rag_query_engine_ask(mock_config_loader, mock_create_engine, mock_sentence_transformer, mock_openai):
    mock_config_loader.load.return_value = MagicMock()
    
    engine = RAGQueryEngine()
    engine._retrieve_context = MagicMock(return_value=[{"a": 1}])
    engine._generate_response = MagicMock(return_value="answer")
    
    res = engine.ask("q", 5)
    assert res == "answer"
    engine._retrieve_context.assert_called_once_with("q", 5)
    engine._generate_response.assert_called_once_with("q", [{"a": 1}])

@patch("services.api.rag_service.OpenAI")
@patch("services.api.rag_service.SentenceTransformer")
@patch("services.api.rag_service.create_engine")
@patch("services.api.rag_service.config_loader")
def test_rag_query_engine_ask_no_context(mock_config_loader, mock_create_engine, mock_sentence_transformer, mock_openai):
    mock_config_loader.load.return_value = MagicMock()
    
    engine = RAGQueryEngine()
    engine._retrieve_context = MagicMock(return_value=[])
    
    res = engine.ask("q", 5)
    assert res == "No data found."
