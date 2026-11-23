"""Predict canopy height using standard or sample-balanced model weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mangrove.inference import parser, predict


if __name__ == "__main__":
    predict(parser().parse_args())
