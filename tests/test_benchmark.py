"""Pure parts of the LLM benchmark (benchmark/benchmark_llm.py): statistics, retries, catalogue snapshot."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import benchmark_llm as bl  # noqa: E402


def _row(model, prompt, seconds, ok=True, valid=True, tokens=100, chars=400):
    row = {"model": model, "prompt": prompt, "run": 1, "ok": ok, "valid": valid and ok}
    if ok:
        row.update(seconds=seconds, completion_tokens=tokens, prompt_tokens=1000, response_chars=chars,
                   tokens_per_second=tokens / seconds)
    return row


def test_stats_ignore_missing_values():
    assert bl._stats([2.0, None, 4.0]) == {"mean": 3.0, "median": 3.0, "min": 2.0, "max": 4.0}
    assert bl._stats([None]) == {"mean": None, "median": None, "min": None, "max": None}


def test_summary_counts_failed_calls_and_valid_answers_per_model_and_prompt():
    rows = [_row("A", "P1", 10), _row("A", "P1", 20, valid=False), _row("A", "P2", 0, ok=False), _row("B", "P1", 5)]
    s = bl.summarise(rows, ["A", "B"], ["P1", "P2"])
    a = s["per_model"]["A"]
    assert (a["calls"], a["ok"], a["failed"], a["valid"]) == (3, 2, 1, 1)
    assert a["seconds"]["mean"] == 15.0
    assert s["per_model_prompt"]["A|P2"]["ok"] == 0
    assert s["per_model_prompt"]["B|P1"]["valid"] == 1


def test_openai_call_retries_when_the_server_is_busy_and_never_exposes_the_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-value")
    monkeypatch.setattr(bl.time, "sleep", lambda s: None)
    busy = MagicMock(status_code=503, ok=False)
    good = MagicMock(status_code=200, ok=True)
    good.json.return_value = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    calls = []

    def post(url, json, headers, timeout):
        calls.append(headers["Authorization"])
        return busy if len(calls) < 3 else good

    monkeypatch.setattr(bl.requests, "post", post)
    result = bl.call_openai({"base_url": "https://x/v1", "model": "m", "api_key_env": "TEST_KEY"}, "hello", 5, 0.2)
    assert result["text"] == "hi" and result["completion_tokens"] == 1 and result["rate_limit_retries"] == 2
    assert len(calls) == 3
    assert "secret-value" not in json.dumps(result)


def test_ollama_call_reads_the_servers_own_token_counts(monkeypatch):
    resp = MagicMock(ok=True)
    resp.json.return_value = {"response": "x", "prompt_eval_count": 12, "eval_count": 7, "eval_duration": 2_000_000_000,
                              "total_duration": 5_000_000_000, "load_duration": 100_000_000}
    monkeypatch.setattr(bl.requests, "post", lambda *a, **k: resp)
    r = bl.call_ollama({"url": "http://h:11434", "model": "m"}, "p", 5, 0.2)
    assert (r["prompt_tokens"], r["completion_tokens"], r["generation_seconds"], r["server_seconds"]) == (12, 7, 2.0, 5.0)


def test_catalogue_snapshot_is_reused_and_not_fetched_again(tmp_path, monkeypatch):
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps({"meta": {"taken_at": "t", "gns3_source": "file"}, "gns3_devices": [{"name": "R"}], "vm_templates": []}))
    monkeypatch.setattr(bl.core, "fetch_available_devices", lambda *a: (_ for _ in ()).throw(AssertionError("fetched")))
    devices, vms, meta = bl.load_catalogs({"gns3_api_url": "http://x"}, {}, snap)
    assert devices == [{"name": "R"}] and vms == [] and meta["gns3_source"] == "file"


def test_catalogue_falls_back_to_the_live_gns3_server_when_the_template_api_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.core, "fetch_available_devices", lambda *a: [])
    monkeypatch.setattr(bl.core, "fetch_nfs_templates", lambda *a: [{"name": "Rocky"}])
    live = MagicMock()
    live.json.return_value = {"templates": [{"name": "Cisco IOSv"}]}
    monkeypatch.setattr(bl.requests, "get", lambda *a, **k: live)
    snap = tmp_path / "s.json"
    devices, vms, meta = bl.load_catalogs({"gns3_api_url": "http://x"}, {"gns3_live": {"host": "10.0.0.1"}}, snap)
    assert devices == [{"name": "Cisco IOSv"}] and "live from GNS3 server 10.0.0.1" in meta["gns3_source"]
    assert snap.exists()
