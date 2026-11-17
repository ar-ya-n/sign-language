"""Download an ASL dataset archive only when a verified URL is supplied.

The project does not bundle or silently invent a dataset URL.  Set
``--url`` to an official release URL and inspect its license before use.
"""
import argparse
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Official archive URL (not guessed by this script).")
    parser.add_argument("--output", type=Path, default=Path("data") / "realsign.zip")
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("No URL supplied. Provide the ASL dataset archive URL explicitly.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(args.url, timeout=120) as source, args.output.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    print(f"Downloaded {args.output}. Verify CC0/license and extract manually before training.")


if __name__ == "__main__":
    main()
