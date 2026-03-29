from __future__ import annotations

import logging
import os

from src.data_collection.store_data import StoreData


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        raise ValueError("RIOT_API_KEY is not set.")

    store = StoreData(api_key=api_key, region="americas", platform_region="na1")
    result = store.store_from_master(match_count=20, delay=1.2)
    print(result)


if __name__ == "__main__":
    main()
