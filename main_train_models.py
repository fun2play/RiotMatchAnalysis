from __future__ import annotations

import logging

from src.modeling.train_models import train_stage_models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metrics_frame = train_stage_models()
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
