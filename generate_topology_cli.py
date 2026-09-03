#!/usr/bin/env python3
"""
CLI zum manuellen Generieren einer Topologie-Config per Ollama LLM.

Beispiel:
    ./venv/bin/python generate_topology_cli.py \\
        --config topology_config.yml \\
        --prompt "Baue eine kleine Topologie mit 2 PCs hinter einem Switch" \\
        --output topology.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from topology_core import generate_topology, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("generate_topology_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generiert eine GNS3-Topologie-Config per Ollama LLM."
    )
    parser.add_argument("--prompt", required=True, help="Beschreibung der gewünschten Topologie")
    parser.add_argument(
        "--config", default="topology_config.yml", help="Pfad zur Config-Datei"
    )
    parser.add_argument("--output", help="Pfad, unter dem die YAML gespeichert werden soll")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    result = generate_topology(args.prompt, config)

    print("\n" + "=" * 60)
    print("Generierte Topologie:")
    print("=" * 60)
    print(result["yaml"])
    print("=" * 60)

    print(f"\nVerwendeter Ollama-Server: {result['ollama_server_used']}")
    print(f"Benötigte Versuche: {len(result['attempts'])}")

    if result["warnings"]:
        print(f"\n{len(result['warnings'])} Warnung(en):")
        for w in result["warnings"]:
            print(f"  - {w}")
    else:
        print("\nKeine Validierungswarnungen. Alle Images sind verfügbar.")

    print(
        f"\nBerücksichtigt: {result['devices_considered']} GNS3-Templates, "
        f"{result['vm_templates_considered']} VM-Images vom NFS-Share."
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result["yaml"], encoding="utf-8")
        print(f"\nGespeichert nach: {output_path}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
