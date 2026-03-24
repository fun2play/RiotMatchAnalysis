from __future__ import annotations

from src.evaluation.evaluate_models import evaluate_saved_models


def main() -> None:
    outputs = evaluate_saved_models()
    print(outputs["contribution"].to_string(index=False))


if __name__ == "__main__":
    main()
