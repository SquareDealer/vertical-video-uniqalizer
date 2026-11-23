"""Versions of one clip for posting from several TikTok accounts.

Every version plays the same cuts and words; they differ in layout and caption style (visible) and
in zoom, color, grain, speed, GOP and container metadata (subtle). No mirroring: it flips on-screen
text. With a webcam overlay no version crops the frame, so the webcam is never lost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import List, Sequence

from .captions import TimedWord


@dataclass
class VariantSpec:
    index: int
    layout: str               # blur | split | smart
    split_top: int            # height of the webcam band in the split layout
    caption_style: str        # karaoke | simple
    caption_color: str
    caption_position: str     # top | middle | bottom
    hook: bool                # open with the best 3 seconds
    zoom: float               # zoom into the game picture (1.0 = none)
    shift: float              # -1..1, horizontal position of the zoom window
    speed: float              # playback speed, pitch kept
    effects: str              # ffmpeg filter chain applied before the captions
    gop: int                  # keyframe interval
    quality: str              # encoder quality preset

    def to_dict(self) -> dict:
        return asdict(self)


# The three looks of the design (docs/superpowers/specs/2026-11-23-s3-watch-uniqualize-design.md).
_LOOKS = (
    dict(caption_style="karaoke", caption_color="#FFE600", caption_position="bottom", hook=False,
         zoom=1.0, shift=0.0, speed=1.0, color="", quality="high"),
    dict(caption_style="karaoke", caption_color="#39FF88", caption_position="middle", hook=True,
         zoom=1.03, shift=-0.6, speed=1.02, color="eq=saturation=1.12:brightness=0.01,colorbalance=rs=0.04:bs=-0.04",
         quality="standard"),
    dict(caption_style="simple", caption_color="#FFFFFF", caption_position="top", hook=False,
         zoom=1.05, shift=0.6, speed=0.98, color="eq=contrast=1.06,colorbalance=rs=-0.03:bs=0.04",
         quality="high"),
)


def variant_presets(count: int, has_facecam: bool) -> List[VariantSpec]:
    specs = []
    for i in range(max(0, count)):
        look, cycle = _LOOKS[i % 3], i // 3
        if i % 3 == 0:
            layout, split_top = ("split", 640) if has_facecam else ("blur", 640)
        elif i % 3 == 1:
            layout, split_top = "blur", 640
        else:
            layout, split_top = ("split", 820) if has_facecam else ("smart", 640)
        shift = look["shift"] + (0.3 * cycle if i % 2 else -0.3 * cycle)
        shift = round(max(-1.0, min(1.0, shift)), 2)
        zoom = look["zoom"] if cycle == 0 else round(look["zoom"] + 0.02 * cycle, 3)
        noise = f"noise=alls={2 + i % 3 + cycle}:allf=t"
        effects = ",".join(part for part in (look["color"], noise) if part)
        specs.append(VariantSpec(
            index=i + 1, layout=layout, split_top=split_top, caption_style=look["caption_style"],
            caption_color=look["caption_color"], caption_position=look["caption_position"], hook=look["hook"],
            zoom=zoom, shift=shift, speed=look["speed"], effects=effects, gop=48 + 12 * i, quality=look["quality"],
        ))
    return specs


def speed_words(words: Sequence[TimedWord], speed: float) -> List[TimedWord]:
    """Caption times of a version played at ``speed``."""
    if speed == 1.0:
        return list(words)
    return [replace(w, start=w.start / speed, end=w.end / speed) for w in words]
