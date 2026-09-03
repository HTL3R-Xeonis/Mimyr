#!/usr/bin/env python3
"""
CLI for manually generating a topology config via an Ollama LLM.

Example:
    ./venv/bin/python generate_topology_cli.py \\
        --config topology_config.yml \\
        --prompt "Build a small topology with 2 PCs behind a switch" \\
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
        description="Generates a GNS3 topology config via an Ollama LLM."
    )
    parser.add_argument("--prompt", required=True, help="Description of the desired topology")
    parser.add_argument(
        "--config", default="topology_config.yml", help="Path to the config file"
    )
    parser.add_argument("--output", help="Path to save the generated YAML to")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    result = generate_topology(args.prompt, config)

    print("\n" + "=" * 60)
    print("Generated topology:")
    print("=" * 60)
    print(result["yaml"])
    print("=" * 60)

    print(f"\nOllama server used: {result['ollama_server_used']}")
    print(f"Attempts needed: {len(result['attempts'])}")

    if result["warnings"]:
        print(f"\n{len(result['warnings'])} warning(s):")
        for w in result["warnings"]:
            print(f"  - {w}")
    else:
        print("\nNo validation warnings. All images are available.")

    print(
        f"\nConsidered: {result['devices_considered']} GNS3 templates, "
        f"{result['vm_templates_considered']} VM images from the NFS share."
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result["yaml"], encoding="utf-8")
        print(f"\nSaved to: {output_path}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
