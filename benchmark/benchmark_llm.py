"""
LLM benchmark for the Topology Generator (diploma goal H8A).

The same prompts - built exactly like the generator builds them, with the real device and
template lists - go to every configured model, several times each. Per call it records the
wall-clock time, the tokens the server counted, the length of the answer and whether the
answer is a valid topology according to the generator's own validation. Results land in a
result folder as raw.jsonl (every call, with the answer), calls.csv, summary.json,
summary.csv and a Markdown report.

Run it where the models and the Template APIs are reachable (the generator container has
everything, including the hosted server's key in its environment):

    python benchmark/benchmark_llm.py --config benchmark/benchmark_config.yml

The API key is only ever read from the environment variable named in `api_key_env` and is
never printed or stored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import topology_core as core  # noqa: E402

WARMUP_PROMPT = "Reply with the single word OK."
RETRY_STATUS = (429, 500, 502, 503, 504)
RETRY_WAITS_SECONDS = (20, 40, 60, 90)   # hosted servers answer 429/503 when busy


def _http_error_text(resp: requests.Response) -> str:
    return f"{resp.status_code} {resp.reason}: {' '.join((resp.text or '').split())[:200]}"


def call_ollama(server: dict, prompt: str, timeout: int, temperature: float, max_tokens: int | None = None) -> dict:
    options = {"temperature": temperature}
    if max_tokens:
        options["num_predict"] = max_tokens
    resp = requests.post(
        f"{server['url'].rstrip('/')}/api/generate",
        json={"model": server["model"], "prompt": prompt, "stream": False, "options": options},
        timeout=(core.CONNECT_TIMEOUT_SECONDS, timeout),
    )
    if not resp.ok:
        raise RuntimeError(_http_error_text(resp))
    data = resp.json()
    eval_seconds = (data.get("eval_duration") or 0) / 1e9
    return {
        "text": data.get("response", ""),
        "thinking_chars": len(data.get("thinking") or ""),
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "reasoning_tokens": None,
        "server_seconds": (data.get("total_duration") or 0) / 1e9 or None,
        "load_seconds": (data.get("load_duration") or 0) / 1e9,
        "generation_seconds": eval_seconds or None,
    }


def call_openai(server: dict, prompt: str, timeout: int, temperature: float, max_tokens: int | None = None) -> dict:
    key_env = server.get("api_key_env")
    headers = {}
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"environment variable {key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": server["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    attempt = 0
    while True:
        resp = requests.post(
            f"{server['base_url'].rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=(core.CONNECT_TIMEOUT_SECONDS, timeout),
        )
        if resp.status_code in RETRY_STATUS and attempt < len(RETRY_WAITS_SECONDS):
            time.sleep(RETRY_WAITS_SECONDS[attempt])
            attempt += 1
            continue
        break
    if not resp.ok:
        raise RuntimeError(_http_error_text(resp))
    data = resp.json()
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return {
        "text": message.get("content") or "",
        "thinking_chars": 0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "server_seconds": None,
        "load_seconds": 0.0,
        "generation_seconds": None,
        "rate_limit_retries": attempt,
    }


def call(server: dict, prompt: str, timeout: int, temperature: float, max_tokens: int | None = None) -> dict:
    fn = call_openai if server.get("type", "ollama") == "openai" else call_ollama
    return fn(server, prompt, timeout, temperature, max_tokens)


def load_catalogs(gen_cfg: dict, cfg: dict, snapshot_path: Path) -> tuple[list[dict], list[dict], dict]:
    """
    The device and VM-template lists every model gets in its prompt. They are frozen in a snapshot file so
    that all models (and every later run) see exactly the same catalogue; without a snapshot they are fetched
    from the Template APIs - and, when the GNS3 Template API has no devices (its collector lost the GNS3
    VM), live from the GNS3 server named under `gns3_live` in the benchmark config.
    """
    if snapshot_path.exists():
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return data["gns3_devices"], data["vm_templates"], data["meta"]
    timeout = gen_cfg.get("api_timeout", 15)
    source = "gns3 template api"
    devices = core.fetch_available_devices(gen_cfg["gns3_api_url"], timeout)
    if not devices and cfg.get("gns3_live"):
        live = cfg["gns3_live"]
        resp = requests.get(
            f"{gen_cfg['gns3_api_url'].rstrip('/')}/api/live-templates",
            params={"host": live["host"], "port": live.get("port", 80), "include_docker_status": "false"},
            timeout=60,
        )
        resp.raise_for_status()
        devices = resp.json().get("templates", [])
        source = f"live from GNS3 server {live['host']}"
    vm_templates = core.fetch_nfs_templates(gen_cfg.get("template_api_url"), timeout)
    meta = {"taken_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "gns3_source": source}
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps({"meta": meta, "gns3_devices": devices, "vm_templates": vm_templates}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return devices, vm_templates, meta


def host_of(server: dict) -> str:
    url = server.get("url") or server.get("base_url")
    return urlparse(url).hostname or url


def measure(server: dict, prompt_id: str, run: int, prompt: str, gns3_devices, vm_templates, cfg: dict) -> dict:
    row = {
        "model": server["label"],
        "prompt": prompt_id,
        "run": run,
        "ok": False,
        "valid": False,
    }
    start = time.perf_counter()
    try:
        result = call(server, prompt, cfg["timeout"], cfg["temperature"], server.get("max_tokens"))
    except Exception as exc:
        row["seconds"] = round(time.perf_counter() - start, 2)
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return row
    row["seconds"] = round(time.perf_counter() - start, 2)
    text = result.pop("text")
    row.update(result)
    row["ok"] = True
    row["response_chars"] = len(text)
    row["response_lines"] = len(text.splitlines())
    row["response"] = text
    if row.get("completion_tokens") and row["seconds"]:
        row["tokens_per_second"] = round(row["completion_tokens"] / row["seconds"], 2)
    fixes: list[str] = []
    try:
        parsed, warnings, _ = core.validate_topology(text, gns3_devices, vm_templates, fixes=fixes)
        row["valid"] = parsed is not None and not warnings
        row["warnings"] = len(warnings)
        row["warning_text"] = "; ".join(warnings[:3])[:300]
        row["auto_fixes"] = len(fixes)
        if parsed:
            row["nodes"] = sum(len(n.get("names", [1])) if isinstance(n, dict) else 1 for n in parsed.get("nodes", []))
    except Exception as exc:
        row["warning_text"] = f"validation crashed: {exc}"[:300]
    return row


def run_server(server: dict, prompts: list[dict], gns3_devices, vm_templates, cfg: dict, sink, log) -> None:
    """All calls to one model, strictly one after the other."""
    key_env = server.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        log(f"[{server['label']}] skipped: environment variable {key_env} is not set")
        return
    log(f"[{server['label']}] warm-up")
    try:
        call(server, WARMUP_PROMPT, cfg["timeout"], cfg["temperature"], max_tokens=16)
    except Exception as exc:
        text = f"{type(exc).__name__}: {str(exc)[:200]}"
        log(f"[{server['label']}] warm-up failed: {text}")
        # no access / unknown model: every later call would fail the same way - record that once and stop
        if any(code in text for code in ("401", "403", "404")):
            sink({"model": server["label"], "prompt": "-", "run": 0, "ok": False, "valid": False, "seconds": 0.0, "error": text})
            return
    for prompt in prompts:
        if server.get("prompts") and prompt["id"] not in server["prompts"]:
            continue
        for run in range(1, cfg["runs"] + 1):
            row = measure(server, prompt["id"], run, prompt["full"], gns3_devices, vm_templates, cfg)
            sink(row)
            state = "ok" if row["ok"] else "FAILED"
            log(
                f"[{server['label']}] {prompt['id']} run {run}: {state} {row.get('seconds')}s "
                f"tokens={row.get('completion_tokens')} valid={row.get('valid')} {row.get('error', '')}"
            )
            if server.get("pause_seconds"):
                time.sleep(server["pause_seconds"])


def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def summarise(rows: list[dict], models: list[str], prompt_ids: list[str]) -> dict:
    def block(selected: list[dict]) -> dict:
        ok = [r for r in selected if r["ok"]]
        return {
            "calls": len(selected),
            "ok": len(ok),
            "failed": len(selected) - len(ok),
            "valid": sum(1 for r in ok if r["valid"]),
            "seconds": _stats([r["seconds"] for r in ok]),
            "prompt_tokens": _stats([r.get("prompt_tokens") for r in ok]),
            "completion_tokens": _stats([r.get("completion_tokens") for r in ok]),
            "response_chars": _stats([r.get("response_chars") for r in ok]),
            "tokens_per_second": _stats([r.get("tokens_per_second") for r in ok]),
        }

    out = {"per_model": {}, "per_model_prompt": {}}
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        out["per_model"][model] = block(mine)
        for pid in prompt_ids:
            out["per_model_prompt"][f"{model}|{pid}"] = block([r for r in mine if r["prompt"] == pid])
    return out


def _fmt(value, suffix="") -> str:
    return "–" if value is None else f"{value}{suffix}"


def write_report(path: Path, started: str, cfg: dict, servers: list[dict], prompts: list[dict], summary: dict, n_dev: int, n_vm: int) -> None:
    lines = [
        "# LLM benchmark (goal H8A)",
        "",
        f"Started: {started}  |  runs per model and prompt: {cfg['runs']}  |  temperature: {cfg['temperature']}  |  "
        f"timeout per call: {cfg['timeout']} s",
        f"Prompt input: the generator's own prompt with {n_dev} GNS3 devices and {n_vm} VM templates.",
        "",
        "## Models",
        "",
        "| Label | Type | Host | Model |",
        "|---|---|---|---|",
    ]
    for s in servers:
        only = f" (only {', '.join(s['prompts'])})" if s.get("prompts") else ""
        lines.append(f"| {s['label']}{only} | {s.get('type', 'ollama')} | {host_of(s)} | {s['model']} |")
    lines += ["", "## Prompts", "", "| ID | Request |", "|---|---|"]
    for p in prompts:
        lines.append(f"| {p['id']} | {p['text']} |")
    lines += [
        "",
        "## Overall per model (successful calls)",
        "",
        "| Model | Calls ok/total | Valid topologies | Time mean (s) | Time median (s) | Time min–max (s) | Output tokens mean | Answer length mean (chars) | Tokens/s mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, b in summary["per_model"].items():
        s = b["seconds"]
        lines.append(
            f"| {label} | {b['ok']}/{b['calls']} | {b['valid']}/{b['ok']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
            f"{_fmt(s['min'])}–{_fmt(s['max'])} | {_fmt(b['completion_tokens']['mean'])} | "
            f"{_fmt(b['response_chars']['mean'])} | {_fmt(b['tokens_per_second']['mean'])} |"
        )
    for title, key in (("Time per prompt (mean, s)", "seconds"), ("Answer length per prompt (mean, chars)", "response_chars"), ("Output tokens per prompt (mean)", "completion_tokens")):
        lines += ["", f"## {title}", "", "| Model | " + " | ".join(p["id"] for p in prompts) + " |", "|---|" + "---|" * len(prompts)]
        for label in summary["per_model"]:
            cells = [_fmt(summary["per_model_prompt"][f"{label}|{p['id']}"][key]["mean"]) for p in prompts]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Valid topologies per prompt (valid/successful calls)", "", "| Model | " + " | ".join(p["id"] for p in prompts) + " |", "|---|" + "---|" * len(prompts)]
    for label in summary["per_model"]:
        cells = []
        for p in prompts:
            b = summary["per_model_prompt"][f"{label}|{p['id']}"]
            cells.append(f"{b['valid']}/{b['ok']}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("benchmark_config.yml")))
    parser.add_argument("--generator-config", default=os.environ.get("CONFIG_PATH", "config/topology_config.yml"))
    parser.add_argument("--out", default="benchmark/results")
    parser.add_argument("--snapshot", default="benchmark/templates_snapshot.json",
                        help="frozen device/template lists; created on the first run, reused afterwards")
    parser.add_argument("--runs", type=int, help="override runs per model and prompt")
    parser.add_argument("--only", help="comma-separated model labels to run (substring match)")
    parser.add_argument("--prompts", help="comma-separated prompt ids, e.g. P1,P2")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.runs:
        cfg["runs"] = args.runs
    servers = cfg["models"]
    if args.only:
        wanted = [w.strip() for w in args.only.split(",")]
        servers = [s for s in servers if any(w in s["label"] for w in wanted)]
    prompts = cfg["prompts"]
    if args.prompts:
        ids = {x.strip() for x in args.prompts.split(",")}
        prompts = [p for p in prompts if p["id"] in ids]

    gen_cfg = core.load_config(args.generator_config)
    gns3_devices, vm_templates, catalog_meta = load_catalogs(gen_cfg, cfg, Path(args.snapshot))
    for p in prompts:
        p["full"] = core.build_prompt(p["text"], gns3_devices, vm_templates, gen_cfg["prompt_template"], gen_cfg["example_topology"])
        p["prompt_chars"] = len(p["full"])

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    rows: list[dict] = []
    raw = open(out_dir / "raw.jsonl", "a", encoding="utf-8")

    def sink(row: dict) -> None:
        with lock:
            rows.append(row)
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()

    def log(msg: str) -> None:
        with lock:
            print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    log(f"{len(servers)} model(s), {len(prompts)} prompt(s), {cfg['runs']} run(s) each; "
        f"{len(gns3_devices)} GNS3 devices, {len(vm_templates)} VM templates; results in {out_dir}")

    by_host: dict[str, list[dict]] = {}
    for s in servers:
        by_host.setdefault(host_of(s), []).append(s)

    def host_worker(group: list[dict]) -> None:
        for s in group:
            run_server(s, prompts, gns3_devices, vm_templates, cfg, sink, log)

    threads = [threading.Thread(target=host_worker, args=(g,)) for g in by_host.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    raw.close()

    labels = [s["label"] for s in servers]
    summary = summarise(rows, labels, [p["id"] for p in prompts])
    summary["meta"] = {
        "started": started,
        "runs": cfg["runs"],
        "temperature": cfg["temperature"],
        "timeout": cfg["timeout"],
        "gns3_devices": len(gns3_devices),
        "vm_templates": len(vm_templates),
        "catalog": catalog_meta,
        "models": [{"label": s["label"], "type": s.get("type", "ollama"), "host": host_of(s), "model": s["model"]} for s in servers],
        "prompts": [{"id": p["id"], "text": p["text"], "prompt_chars": p["prompt_chars"]} for p in prompts],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["model", "prompt", "run", "ok", "valid", "seconds", "prompt_tokens", "completion_tokens", "reasoning_tokens",
               "response_chars", "response_lines", "tokens_per_second", "warnings", "auto_fixes", "nodes", "load_seconds", "error"]
    with open(out_dir / "calls.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "calls", "ok", "valid", "time_mean_s", "time_median_s", "time_min_s", "time_max_s",
                         "output_tokens_mean", "answer_chars_mean", "tokens_per_s_mean"])
        for label, b in summary["per_model"].items():
            s = b["seconds"]
            writer.writerow([label, b["calls"], b["ok"], b["valid"], s["mean"], s["median"], s["min"], s["max"],
                             b["completion_tokens"]["mean"], b["response_chars"]["mean"], b["tokens_per_second"]["mean"]])
    write_report(out_dir / "report.md", started, cfg, servers, prompts, summary, len(gns3_devices), len(vm_templates))
    log(f"done - report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
