"""``python -m ave_emotion.selftest``: runs at image build, so a wrong adapter or other labels break the build."""

import sys

import numpy as np

from .audio import SAMPLE_RATE
from .model import load_gigaam


def main() -> int:
    model = load_gigaam()
    noise = (np.random.default_rng(0).normal(0.0, 0.05, SAMPLE_RATE * 3)).astype(np.float32)
    probs = model.probs(noise)
    total = sum(probs.values())
    labels = {label.lower() for label in model.labels}
    if abs(total - 1.0) > 0.01:
        print(f"selftest: probabilities sum to {total:.3f}, not 1", file=sys.stderr)
        return 1
    if not {"positive", "angry"} <= labels:
        print(f"selftest: labels {sorted(labels)} lack positive/angry", file=sys.stderr)
        return 1
    print(f"selftest ok: {', '.join(f'{k}={v:.2f}' for k, v in probs.items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
