"""CLI entry: ``python -m opennta.analysis.unet_field.unet_bundle <spots.csv>``."""

import argparse
import json
import logging

from .config import OUTPUTS_DIR
from .inference import UNetInferenceRunner

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run U-Net actual inference for one spots.csv file")
    parser.add_argument("spots_input", help="Path to input spots.csv")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR), help="Directory to save outputs")
    args = parser.parse_args()

    runner = UNetInferenceRunner()
    result = runner.run_and_save_outputs(args.spots_input, output_dir=args.output_dir)
    logger.info("inference summary:\n%s", json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
