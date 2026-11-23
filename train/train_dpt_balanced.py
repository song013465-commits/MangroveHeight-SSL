"""Train the balanced canopy height model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mangrove.training import parser, train_model


if __name__ == "__main__":
    train_model(parser("balanced").parse_args())
