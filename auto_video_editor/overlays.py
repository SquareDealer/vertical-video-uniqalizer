"""Branding overlays of the TikTok versions: a divider on the webcam/game seam or a banner.

The assets (data/overlays, described by its overlay.json) are 1080 px wide, transparent and come as an
animated WebM (VP9 with alpha) and a still PNG. ``neon`` and ``ticker`` sit on a seam and loop for the
whole clip; ``banner`` has a fixed place and plays once, ``BANNER_START`` seconds into the clip.

ffmpeg's built-in ``vp9`` decoder drops the alpha channel (the glow becomes a solid fill), so the WebM is
read with ``libvpx-vp9``. A build without it gets the PNG instead: the same picture, not animated.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .ffmpeg import ffmpeg_bin, spawn_options

KINDS = ("neon", "ticker", "banner")
BANNER_START = 2.0   # seconds: the banner slides in once the viewer is watching
MIN_SEAM = 300       # a seam closer to the top is under the TikTok header: no divider there
FPS = 30


@dataclass(frozen=True)
class Overlay:
    kind: str
    path: str        # .webm (animated) or .png (still)
    y: int           # top edge in the 1080x1920 frame
    height: int
    loop: bool       # repeat for the whole clip (dividers) or play once (banner)
    start: float     # seconds before a one-shot overlay appears

    @property
    def animated(self) -> bool:
        return self.path.lower().endswith(".webm")

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def input_args(self) -> List[str]:
        """ffmpeg arguments of the overlay input, placed after the inputs of the clip."""
        if self.animated:
            args = ["-stream_loop", "-1"] if self.loop else []
            if self.start > 0:
                args += ["-itsoffset", f"{self.start:g}"]
            return [*args, "-c:v", "libvpx-vp9", "-i", self.path]
        return ["-loop", "1", "-framerate", str(FPS), "-i", self.path]

    def filter(self, src: str, index: int, dst: str) -> str:
        """``src`` with the overlay (ffmpeg input ``index``) drawn on it, labelled ``dst``.

        The output length is set by the render's ``-t``; ``shortest`` ends an endless loop with the clip.
        """
        if self.loop:
            return f"{src}[{index}:v]overlay=0:{self.y}:shortest=1{dst}"
        if self.animated:
            # The banner input starts late (-itsoffset) and ends after one play: nothing is drawn before or after.
            return f"{src}[{index}:v]overlay=0:{self.y}:eof_action=pass{dst}"
        end = self.start + _duration(Path(self.path).parent, self.kind)
        return (f"{src}[{index}:v]overlay=0:{self.y}:shortest=1"
                f":enable='between(t,{self.start:g},{end:g})'{dst}")


def default_dir() -> Path:
    return Path(__file__).parent / "data" / "overlays"


@lru_cache(maxsize=8)
def _config(directory: str) -> dict:
    return json.loads((Path(directory) / "overlay.json").read_text(encoding="utf-8"))


def _asset(directory: Path, kind: str) -> dict:
    assets = _config(str(directory)).get("assets", {})
    if kind not in assets:
        raise ValueError(f"overlay '{kind}' is not described in {directory / 'overlay.json'}")
    return assets[kind]


def _duration(directory: Path, kind: str) -> float:
    return float(_asset(directory, kind).get("durationSec", 5))


def check_dir(directory: str) -> Optional[str]:
    """Error message when ``directory`` is not a usable overlay pack, else None."""
    path = Path(directory)
    try:
        config = _config(str(path))
    except (OSError, ValueError) as exc:
        return f"overlay folder {path}: overlay.json cannot be read ({exc})"
    for kind in KINDS:
        asset = config.get("assets", {}).get(kind)
        if not asset:
            return f"overlay folder {path}: overlay.json has no '{kind}'"
        if not any((path / asset.get(key, "")).is_file() for key in ("webm", "png") if asset.get(key)):
            return f"overlay folder {path}: no file for '{kind}'"
    return None


@lru_cache(maxsize=1)
def alpha_decoder() -> bool:
    """True when ffmpeg can decode VP9 with its alpha channel (libvpx-vp9)."""
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-decoders"], capture_output=True, text=True,
                             timeout=30, **spawn_options()).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == ["libvpx-vp9"] for line in out.splitlines())


def place(kind: str, seam: Optional[int], directory: Optional[str] = None,
          animated: Optional[bool] = None) -> Optional[Overlay]:
    """Overlay ``kind`` placed in the frame; ``seam`` is the y of the edge between two pictures
    (webcam/game of the split layout, blurred band/picture of the blur layout), None when there is none.

    A divider without a seam turns into the banner: it has a place of its own in every layout.
    """
    if not kind:
        return None
    if kind not in KINDS:
        raise ValueError(f"unknown overlay '{kind}' (expected one of {', '.join(KINDS)})")
    folder = Path(directory) if directory else default_dir()
    if _asset(folder, kind).get("anchor") == "seam" and (seam is None or seam < MIN_SEAM):
        kind = "banner"
    asset = _asset(folder, kind)
    height = int(asset["height"])
    if asset.get("anchor") == "seam":
        y = int(seam) - height // 2
    else:
        y = int(asset.get("y", 1130))
    use_webm = alpha_decoder() if animated is None else animated
    name = asset.get("webm") if use_webm and asset.get("webm") else asset.get("png") or asset.get("webm")
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"overlay file not found: {path}")
    loop = bool(asset.get("loop", asset.get("anchor") == "seam"))
    return Overlay(kind=kind, path=str(path.resolve()), y=y, height=height, loop=loop,
                   start=0.0 if loop else BANNER_START)
