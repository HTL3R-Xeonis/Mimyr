# LLM benchmark (diploma goal H8A)

`benchmark_llm.py` sends the same prompts - built exactly like the generator builds them - to every model in
`benchmark_config.yml` and records, per call, the wall-clock time, the tokens the server counted, the length of
the answer and whether the answer is a valid topology according to the generator's own validation.

```bash
# inside the generator container (it has the Template-API access and the key of the hosted server)
docker exec -w /app topology-generator python benchmark/benchmark_llm.py \
    --generator-config /config/topology_config.yml --snapshot benchmark/templates_snapshot.json
```

* `templates_snapshot.json` freezes the device and VM-template lists, so every model (and every later run) gets
  the same catalogue. If the file does not exist it is created from the Template APIs (or, when the GNS3
  Template API has no devices, live from the GNS3 server named under `gns3_live`).
* The API key of a hosted server is read only from the environment variable named in `api_key_env`; it is never
  printed or stored. Busy servers (HTTP 429/5xx) are retried with growing waits.
* Options: `--runs N`, `--only "<label substring>,..."`, `--prompts P1,P2`, `--out DIR`.
* Results per run in `benchmark/results/<timestamp>/` (git-ignored): `raw.jsonl` (every call incl. the answer),
  `calls.csv`, `summary.csv`, `summary.json`, `report.md`.
