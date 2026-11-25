"""The emotion model behind the service. ``torch`` and ``gigaam`` are imported only when it is loaded."""

import os
from typing import Dict, List

try:
    from typing import Protocol
except ImportError:  # pragma: no cover - Python 3.7
    Protocol = object  # type: ignore[assignment,misc]

import numpy as np


class EmotionModel(Protocol):
    labels: List[str]

    def probs(self, samples: np.ndarray) -> Dict[str, float]:
        """Probabilities of every label for one window of 16 kHz mono float samples."""


class GigaAMEmotion:
    """Adapter over ``GigaAMEmo``: the upstream ``get_probs`` only takes a file path, this takes samples."""

    def __init__(self, model) -> None:
        self._model = model
        self.labels = [str(model.id2name[i]) for i in range(len(model.id2name))]

    def probs(self, samples: np.ndarray) -> Dict[str, float]:
        import torch

        model = self._model
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to(model._device).to(model._dtype)
        wav = wav.unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=model._device)
        with torch.inference_mode():
            encoded, _ = model.forward(wav, length)
            pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
            values = torch.nn.functional.softmax(model.head(pooled)[0], dim=-1).tolist()
        return {name: float(values[i]) for i, name in enumerate(self.labels)}


def load_gigaam() -> EmotionModel:
    """GigaAM-Emo on the CPU with a fixed number of threads (the service shares the machine)."""
    import gigaam
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return GigaAMEmotion(gigaam.load_model("emo", device="cpu"))
