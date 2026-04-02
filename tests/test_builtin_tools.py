"""Tests for built-in tools."""

from unittest.mock import MagicMock, patch

import pytest


# ── Calculator ────────────────────────────────────────────────────────────────

from agent_system.tools.builtin_tools import calculate, get_datetime, summarise_text


def test_calculate_basic_arithmetic():
    assert calculate.invoke("2 + 2") == "4"
    assert calculate.invoke("10 - 3") == "7"
    assert calculate.invoke("6 * 7") == "42"
    assert calculate.invoke("10 / 4") == "2.5"
    assert calculate.invoke("2 ** 10") == "1024"


def test_calculate_math_functions():
    assert calculate.invoke("sqrt(16)") == "4"
    assert calculate.invoke("factorial(5)") == "120"
    assert calculate.invoke("abs(-42)") == "42"
    assert calculate.invoke("round(3.7)") == "4"
    assert calculate.invoke("max(1, 5, 3)") == "5"


def test_calculate_constants():
    result = calculate.invoke("pi")
    assert result.startswith("3.14")

    result = calculate.invoke("e")
    assert result.startswith("2.71")


def test_calculate_division_by_zero():
    result = calculate.invoke("1 / 0")
    assert "division by zero" in result.lower()


def test_calculate_blocks_unsafe_code():
    result = calculate.invoke("__import__('os').system('ls')")
    assert "Error" in result or "error" in result.lower()


def test_calculate_unknown_function_blocked():
    result = calculate.invoke("open('/etc/passwd').read()")
    assert "Error" in result or "error" in result.lower()


def test_calculate_returns_int_for_whole_floats():
    assert calculate.invoke("4.0") == "4"
    assert calculate.invoke("10 / 2") == "5"


# ── get_datetime ──────────────────────────────────────────────────────────────

def test_get_datetime_returns_utc_string():
    result = get_datetime.invoke("UTC")
    assert "UTC" in result
    assert "202" in result  # year prefix


# ── summarise_text ────────────────────────────────────────────────────────────

def test_summarise_text_short_passthrough():
    text = "Hello world."
    assert summarise_text.invoke({"text": text, "max_chars": 2000}) == text


def test_summarise_text_truncates_long_text():
    text = "A" * 5000
    result = summarise_text.invoke({"text": text, "max_chars": 100})
    assert len(result) < 5000
    assert "more characters" in result


def test_summarise_text_breaks_at_sentence():
    text = "First sentence. " + "B" * 3000
    result = summarise_text.invoke({"text": text, "max_chars": 500})
    # Should prefer to break at the period
    assert result.startswith("First sentence.")


# ── fetch_url ─────────────────────────────────────────────────────────────────

from agent_system.tools.builtin_tools import fetch_url


def test_fetch_url_rejects_non_http():
    result = fetch_url.invoke("ftp://example.com/file")
    assert "Error" in result


def test_fetch_url_success(respx_mock=None):
    """Test fetch_url with a mocked HTTP response."""
    with patch("httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.text = "<html>Hello from mock</html>"
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = fetch_url.invoke({"url": "https://example.com", "timeout": 5})
        assert "Hello from mock" in result


# ── write_file / read_file / list_files ──────────────────────────────────────

from agent_system.tools.builtin_tools import list_files, read_file, write_file


def _mock_minio(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(
        "agent_system.tools.builtin_tools._get_minio", lambda: mock
    )
    return mock


def test_write_file_success(monkeypatch):
    mock = _mock_minio(monkeypatch)
    result = write_file.invoke({"path": "test/file.txt", "content": "hello"})
    mock.upload_bytes.assert_called_once()
    assert "test/file.txt" in result


def test_read_file_success(monkeypatch):
    mock = _mock_minio(monkeypatch)
    mock.download_bytes.return_value = b"file content here"
    result = read_file.invoke("test/file.txt")
    assert "file content here" in result


def test_list_files_empty(monkeypatch):
    mock = _mock_minio(monkeypatch)
    mock.list_objects.return_value = []
    result = list_files.invoke("")
    assert "No files" in result


def test_list_files_with_results(monkeypatch):
    mock = _mock_minio(monkeypatch)
    mock.list_objects.return_value = ["reports/a.txt", "reports/b.txt"]
    result = list_files.invoke("reports/")
    assert "reports/a.txt" in result
    assert "reports/b.txt" in result


# ── web_search (mocked — no real network) ────────────────────────────────────

from agent_system.tools.builtin_tools import web_search


def test_web_search_returns_results():
    mock_results = [
        {"title": "LangGraph Docs", "href": "https://langchain.com", "body": "LangGraph is..."},
        {"title": "Another Result", "href": "https://example.com", "body": "Some snippet."},
    ]
    with patch("duckduckgo_search.DDGS") as mock_ddgs_cls:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=mock_results)
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search.invoke({"query": "LangGraph", "max_results": 2})
        assert "LangGraph Docs" in result
        assert "https://langchain.com" in result


def test_web_search_no_results():
    with patch("duckduckgo_search.DDGS") as mock_ddgs_cls:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=[])
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search.invoke({"query": "xyzzy_nonexistent_1234", "max_results": 3})
        assert "No results" in result
