"""Command-line entry point.

Defaults to the recommended solution. The comparison between solutions is the
*evidence* in the report; this CLI is the *deliverable* -- one clip in, one
sentiment out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssa", description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="path to an audio file")
    parser.add_argument(
        "--solution",
        default="fusion",
        choices=["lexical", "acoustic", "fusion"],
        help="which solution to run (default: the recommended one)",
    )
    parser.add_argument(
        "--backend",
        default="permissive",
        choices=["permissive", "research"],
        help="acoustic backend; 'research' is CC-BY-NC-SA-4.0, research use only",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.parse_args(argv)

    raise SystemExit("not implemented yet -- see docs/TASKS.md")


if __name__ == "__main__":
    sys.exit(main())
