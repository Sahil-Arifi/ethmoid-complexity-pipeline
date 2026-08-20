from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ethmoid-app")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="Start the local ethmoid application")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)
    start.add_argument("--no-browser", action="store_true")
    start.add_argument("--work-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        import uvicorn

        from . import create_app

        url = f"http://{args.host}:{args.port}"
        if not args.no_browser:
            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        uvicorn.run(create_app(work_root=args.work_root), host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
