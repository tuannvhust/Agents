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


@pytest.mark.asyncio
async def test_write_file_scopes_minio_key_with_run_context(monkeypatch):
    from agent_system.core.run_context import RunContext, run_ctx

    mock = _mock_minio(monkeypatch)
    token = run_ctx.set(RunContext(run_id="session-xyz", agent_name="researcher"))
    try:
        await write_file.ainvoke({"path": "reports/out.txt", "content": "data"})
    finally:
        run_ctx.reset(token)
    mock.upload_bytes.assert_called_once()
    key, payload = mock.upload_bytes.call_args[0]
    assert key == "runs/researcher/session-xyz/reports/out.txt"
    assert payload == b"data"


@pytest.mark.asyncio
async def test_list_files_scopes_prefix_with_run_context(monkeypatch):
    from agent_system.core.run_context import RunContext, run_ctx

    mock = _mock_minio(monkeypatch)
    mock.list_objects.return_value = ["runs/ag/r1/reports/a.txt"]
    token = run_ctx.set(RunContext(run_id="r1", agent_name="ag"))
    try:
        result = list_files.invoke("reports/")
    finally:
        run_ctx.reset(token)
    mock.list_objects.assert_called_once_with(prefix="runs/ag/r1/reports/")
    assert "a.txt" in result


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


# ── OCR gateway (mocked HTTP) ───────────────────────────────────────────────

from agent_system.tools.builtin_tools import ocr_document, ocr_get_job, ocr_minio_document


def test_ocr_document_not_configured():
    mock_cfg = MagicMock()
    mock_cfg.url = ""
    mock_cfg.api_key = ""
    mock_cfg.poll_interval = 5.0
    mock_cfg.max_wait_seconds = 600.0
    with patch("agent_system.tools.builtin_tools._ocr_gateway_config", return_value=mock_cfg):
        result = ocr_document.invoke({"file_path": "/tmp/x.pdf", "doc_type": "t"})
    assert "not configured" in result.lower()


def test_ocr_document_file_missing():
    mock_cfg = MagicMock()
    mock_cfg.url = "http://localhost:9999/api/v1/job"
    mock_cfg.api_key = "k"
    mock_cfg.poll_interval = 0.5
    mock_cfg.max_wait_seconds = 10.0
    with patch("agent_system.tools.builtin_tools._ocr_gateway_config", return_value=mock_cfg):
        result = ocr_document.invoke(
            {"file_path": "/nonexistent/path/ocr_test_file.pdf", "doc_type": "t"}
        )
    assert "not found" in result.lower() or "Error" in result


def test_ocr_document_success(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")

    mock_cfg = MagicMock()
    mock_cfg.url = "http://localhost:9999/api/v1/job"
    mock_cfg.api_key = "secret"
    mock_cfg.poll_interval = 0.01
    mock_cfg.max_wait_seconds = 5.0

    post_resp = MagicMock()
    post_resp.json.return_value = {"job_ckey": "ck1-abc", "status": "QUEUED"}
    post_resp.raise_for_status = MagicMock()

    get_resp_queued = MagicMock()
    get_resp_queued.json.return_value = {"job_ckey": "ck1-abc", "status": "QUEUED"}
    get_resp_queued.raise_for_status = MagicMock()

    get_resp_done = MagicMock()
    inner = {"title": "Doc title", "fee_type": "SHA"}
    get_resp_done.json.return_value = {
        "job_ckey": "ck1-abc",
        "status": "COMPLETED",
        "result": {"result": inner, "success": True},
    }
    get_resp_done.raise_for_status = MagicMock()

    call_count = {"n": 0}

    def mock_get(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] < 2:
            return get_resp_queued
        return get_resp_done

    with patch("agent_system.tools.builtin_tools._ocr_gateway_config", return_value=mock_cfg):
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post = MagicMock(return_value=post_resp)
            mock_client.get = MagicMock(side_effect=mock_get)
            mock_client_cls.return_value = mock_client

            result = ocr_document.invoke(
                {
                    "file_path": str(pdf),
                    "doc_type": "gdnct_khcn_ttqt",
                    "poll_interval_seconds": 0.01,
                }
            )

    assert "Doc title" in result
    assert "fee_type" in result
    mock_client.post.assert_called_once()


def test_ocr_get_job_completed():
    mock_cfg = MagicMock()
    mock_cfg.url = "http://localhost:9999/api/v1/job"
    mock_cfg.api_key = "k"

    get_resp = MagicMock()
    get_resp.json.return_value = {
        "status": "COMPLETED",
        "result": {"result": {"x": 1}},
    }
    get_resp.raise_for_status = MagicMock()

    with patch("agent_system.tools.builtin_tools._ocr_gateway_config", return_value=mock_cfg):
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get = MagicMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = ocr_get_job.invoke({"job_ckey": "ck1-abc"})

    assert "1" in result and "x" in result
    mock_client.get.assert_called_once()


def test_ocr_minio_document_success():
    mock_cfg = MagicMock()
    mock_cfg.url = "http://localhost:9999/api/v1/job"
    mock_cfg.api_key = "secret"
    mock_cfg.poll_interval = 0.01
    mock_cfg.max_wait_seconds = 5.0

    minio_mock = MagicMock()
    minio_mock.download_bytes.return_value = b"%PDF-1.4"

    post_resp = MagicMock()
    post_resp.json.return_value = {"job_ckey": "ck-minio", "status": "QUEUED"}
    post_resp.raise_for_status = MagicMock()

    get_resp_done = MagicMock()
    get_resp_done.json.return_value = {
        "job_ckey": "ck-minio",
        "status": "COMPLETED",
        "result": {"result": {"from": "minio"}, "success": True},
    }
    get_resp_done.raise_for_status = MagicMock()

    with patch("agent_system.tools.builtin_tools._ocr_gateway_config", return_value=mock_cfg):
        with patch("agent_system.tools.builtin_tools._get_minio", return_value=minio_mock):
            with patch("httpx.Client") as mock_client_cls:
                mock_http = MagicMock()
                mock_http.__enter__ = MagicMock(return_value=mock_http)
                mock_http.__exit__ = MagicMock(return_value=False)
                mock_http.post = MagicMock(return_value=post_resp)
                mock_http.get = MagicMock(return_value=get_resp_done)
                mock_client_cls.return_value = mock_http

                result = ocr_minio_document.invoke(
                    {
                        "path": "scans/doc.pdf",
                        "doc_type": "gdnct_khcn_ttqt",
                        "poll_interval_seconds": 0.01,
                    }
                )

    assert "minio" in result
    minio_mock.download_bytes.assert_called_once_with("scans/doc.pdf")
    mock_http.post.assert_called_once()
