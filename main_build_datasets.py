from __future__ import annotations

from src.preprocessing.make_datasets import build_datasets


def main() -> None:
    result = build_datasets()
    print(f"Built {result['rows']} stage rows from {result['matches']} matches")


if __name__ == "__main__":
    main()
