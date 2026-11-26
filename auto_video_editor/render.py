"""Building ffmpeg filter graphs and rendering clips, reels and vertical shorts."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .ffmpeg import MediaInfo, audio_codec_args, audio_input, run_ffmpeg, video_codec_args
from .overlays import Overlay
from .timeline import Piece, as_piece, pieces_duration

VERTICAL_W, VERTICAL_H = 1080, 1920
SPLIT_TOP_H = 640
MICRO_FADE = 0.02  # seconds; removes clicks at every audio cut
SEEK_PREROLL = 3.0  # seconds decoded before each cut, then trimmed away


def fps_expression(fps: float) -> str:
    for base in (24, 30, 60, 120):
        if abs(fps - base * 1000 / 1001) < 0.005:
            return f"{base * 1000}/1001"
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


# -- vertical layouts ---------------------------------------------------------------------------

def zoom_window(width: int, height: int, zoom: float, shift: float,
                keep: Optional[Tuple[float, float, float, float]] = None) -> Tuple[int, int, int, int]:
    """Crop ``(w, h, x, y)`` that zooms into a ``width``x``height`` picture by ``zoom``.

    ``shift`` (-1..1) moves the window horizontally across the free margin. ``keep`` is a normalized
    rectangle (a webcam overlay) that must stay fully inside the window.
    """
    if zoom <= 1.0:
        return width, height, 0, 0
    w, h = min(width, _even(width / zoom)), min(height, _even(height / zoom))
    free_x, free_y = width - w, height - h
    shift = max(-1.0, min(1.0, shift))
    x, y = free_x * (0.5 + shift / 2), free_y / 2
    if keep:
        kx, ky, kw, kh = keep
        left, right = kx * width, (kx + kw) * width
        top, bottom = ky * height, (ky + kh) * height
        if right - left <= w:
            x = min(max(x, right - w), left)
        if bottom - top <= h:
            y = min(max(y, bottom - h), top)
    x = int(min(max(0.0, x), free_x)) // 2 * 2
    y = int(min(max(0.0, y), free_y)) // 2 * 2
    if keep:  # rounding down to even pixels must not push the window off the far edge of the overlay
        if (keep[0] + keep[2]) * width > x + w and x + 2 <= free_x:
            x += 2
        if (keep[1] + keep[3]) * height > y + h and y + 2 <= free_y:
            y += 2
    return w, h, x, y


def _zoom_crop(width: int, height: int, zoom: float, shift: float,
               keep: Optional[Tuple[float, float, float, float]] = None) -> str:
    if zoom <= 1.0:
        return ""
    w, h, x, y = zoom_window(width, height, zoom, shift, keep)
    return f"crop={w}:{h}:{x}:{y},"


def _area_crop(width: int, height: int, area: Optional[Tuple[float, float, float, float]]) -> Tuple[str, int, int]:
    """Crop of the stream area (the game picture drawn in the Swiper): filter prefix and its size."""
    if not area:
        return "", width, height
    x, y, w, h = area
    cw, ch = min(width, _even(w * width)), min(height, _even(h * height))
    cx = min(max(0, int(round(x * width))), width - cw)
    cy = min(max(0, int(round(y * height))), height - ch)
    return f"crop={cw}:{ch}:{cx}:{cy},", cw, ch


def vertical_filter(src: str, dst: str, info: MediaInfo, layout: str,
                    subject_x: Optional[float] = None,
                    facecam: Optional[Tuple[float, float, float, float]] = None, blur: int = 10,
                    game_zoom: float = 1.0, game_shift: float = 0.0, split_top: int = SPLIT_TOP_H,
                    game_area: Optional[Tuple[float, float, float, float]] = None) -> str:
    """Filter graph that turns ``src`` into a 1080x1920 frame labelled ``dst``.

    ``game_area`` (x, y, w, h fractions) is the game picture of the blur and split layouts; without it
    the whole frame is used.

    ``game_zoom``/``game_shift`` zoom into the game picture only: the webcam of the split layout is
    left as is, and in the blur layout the zoom never moves ``facecam`` out of frame.
    """
    W, H = VERTICAL_W, VERTICAL_H
    src_w, src_h = info.width or 1920, info.height or 1080
    area, area_w, area_h = _area_crop(src_w, src_h, game_area)
    if layout == "blur":
        radius = max(1, min(60, int(blur)))
        return (
            f"{src}split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"boxblur={radius}:2,scale={W}:{H},eq=brightness=-0.06:saturation=1.15[bg];"
            f"[fgsrc]{area}{_zoom_crop(area_w, area_h, game_zoom, game_shift, None if game_area else facecam)}"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{dst}"
        )
    if layout == "split" and facecam:
        fx, fy, fw, fh = facecam
        cw, ch = _even(fw * info.width), _even(fh * info.height)
        cx = min(max(0, int(fx * info.width)), max(0, info.width - cw))
        cy = min(max(0, int(fy * info.height)), max(0, info.height - ch))
        top = max(2, min(H - 2, _even(split_top)))
        bottom = H - top
        return (
            f"{src}split=2[camsrc][gamesrc];"
            f"[camsrc]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top}:force_original_aspect_ratio=increase,"
            f"crop={W}:{top}[cam];"
            f"[gamesrc]{area}{_zoom_crop(area_w, area_h, game_zoom, game_shift)}"
            f"scale={W}:{bottom}:force_original_aspect_ratio=increase,crop={W}:{bottom}[game];"
            f"[cam][game]vstack=inputs=2,setsar=1{dst}"
        )
    # crop / smart crop
    if src_w / src_h > W / H:
        cw, ch = min(src_w, _even(src_h * W / H)), src_h
        center = (subject_x if subject_x is not None else 0.5) * src_w
        x, y = int(min(max(0.0, center - cw / 2), src_w - cw)), 0
    else:
        cw, ch = src_w, min(src_h, _even(src_w * H / W))
        x, y = 0, int((src_h - ch) / 2)
    if game_zoom > 1.0:
        zw, zh, zx, zy = zoom_window(cw, ch, game_zoom, game_shift)
        cw, ch, x, y = zw, zh, x + zx, y + zy
    return f"{src}crop={cw}:{ch}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1{dst}"


def seam_y(layout: str, info: MediaInfo, split_top: int = SPLIT_TOP_H,
           game_area: Optional[Tuple[float, float, float, float]] = None) -> Optional[int]:
    """Height of the edge between two pictures of a vertical layout: the webcam band and the game
    (split), the blurred band and the picture (blur). None for a full-frame crop or a picture that
    fills the frame."""
    if layout == "split":
        return max(2, min(VERTICAL_H - 2, _even(split_top)))
    if layout != "blur":
        return None
    _crop, w, h = _area_crop(info.width or 1920, info.height or 1080, game_area)
    if w * VERTICAL_H <= h * VERTICAL_W:  # as tall as the frame or taller: no band above the picture
        return None
    return (VERTICAL_H - _even(VERTICAL_W * h / w)) // 2


def caption_top_margin(layout: str, split_top: int = SPLIT_TOP_H,
                       overlay: Optional[Overlay] = None) -> Optional[int]:
    """Top margin for captions placed at the top: below the webcam band of the split layout, so they
    never cover the streamer's face, and below a divider drawn on its seam. None = the usual top margin."""
    if layout != "split":
        return None
    margin = split_top + 40
    if overlay is not None and overlay.y <= split_top <= overlay.bottom:
        margin = max(margin, overlay.bottom)
    return margin


def caption_margin(layout: str, info: MediaInfo) -> int:
    """Vertical margin (from the bottom) that keeps captions off the busy parts of the frame."""
    if layout == "split":
        return VERTICAL_H - SPLIT_TOP_H - 190
    if layout == "blur":
        video_h = VERTICAL_W / (info.aspect or 16 / 9)
        return max(260, int((VERTICAL_H - video_h) / 2 - 190))
    return int(VERTICAL_H * 0.28)


# -- loudness ---------------------------------------------------------------------------------------

def loudnorm_measure_filter(target_lufs: float) -> str:
    return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json"


def parse_loudnorm(stderr: str) -> Optional[dict]:
    matches = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
    if not matches:
        return None
    try:
        data = json.loads(matches[-1])
        float(data["input_i"])
        if data["input_i"] in ("-inf", "inf"):
            return None
        return data
    except (ValueError, KeyError):
        return None


def loudnorm_apply_filter(measured: Optional[dict], target_lufs: float) -> str:
    if not measured:
        return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    return (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:measured_I={measured['input_i']}"
        f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true"
    )


# -- renderer -----------------------------------------------------------------------------------------

class Renderer:
    def __init__(self, info: MediaInfo, encoder: str, quality: str,
                 cancel_event: Optional[threading.Event] = None, audio_bitrate: int = 192,
                 blur: int = 10, audio_track: int = 1) -> None:
        self.info = info
        self.encoder = encoder
        self.quality = quality
        self.cancel_event = cancel_event
        self.audio_args = audio_codec_args(audio_bitrate)
        self.blur = blur
        self.audio_track = audio_track  # audio track of the source heard in the output (1-based, 0 = all mixed)

    def build_segment_command(
        self,
        ranges: Sequence[Sequence[float]],
        output: str,
        *,
        layout: Optional[str] = None,
        subject_x: Optional[float] = None,
        facecam: Optional[Tuple[float, float, float, float]] = None,
        captions_file: Optional[str] = None,
        fade: float = 0.0,
        audio_filter: Optional[str] = None,
        speed: float = 1.0,
        game_zoom: float = 1.0,
        game_shift: float = 0.0,
        split_top: int = SPLIT_TOP_H,
        game_area: Optional[Tuple[float, float, float, float]] = None,
        video_effects: Optional[str] = None,
        overlay: Optional[Overlay] = None,
        extra_output_args: Sequence[str] = (),
    ) -> List[str]:
        """ffmpeg arguments that cut ``ranges`` out of the source into one file.

        A range is ``(start, end)`` or ``(start, end, speed)``: a piece with its own speed (0.5..4.0) is
        sped up or slowed down before the pieces are joined. ``speed`` plays the result faster or slower
        without changing the pitch; ``video_effects`` is an ffmpeg filter chain applied before the captions;
        ``overlay`` (a divider or a banner) is drawn on the vertical layout, under the captions;
        ``extra_output_args`` go just before the output.
        """
        info = self.info
        pieces = [as_piece(r) for r in ranges]
        pieces = [p for p in pieces if p[1] - p[0] > 0.04]
        if not pieces:
            raise ValueError("Nothing to render: empty time range")
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"speed {speed} is outside 0.5..2.0")
        for _, _, piece_speed in pieces:
            if not 0.5 <= piece_speed <= 4.0:
                raise ValueError(f"piece speed {piece_speed} is outside 0.5..4.0")
        out_duration = pieces_duration(pieces)

        # Chronological ranges share one input. A range that goes back in time (a teaser hook) opens a
        # new input, so ffmpeg never has to keep minutes of decoded frames in memory.
        runs: List[List[Piece]] = []
        for rg in pieces:
            if runs and rg[0] >= runs[-1][-1][1] - 1e-3:
                runs[-1].append(rg)
            else:
                runs.append([rg])
        args: List[str] = []
        graph: List[str] = []
        vsrc: List[str] = []
        asrc: List[str] = []
        local: List[Piece] = []
        for index, run in enumerate(runs):
            # Seek a little before the cut: fast seeking can land after the target (MPEG-TS) and the
            # frame shown at the cut point may have been decoded before it (variable frame rate).
            origin, finish = max(0.0, run[0][0] - SEEK_PREROLL), run[-1][1]
            args += ["-ss", f"{origin:.3f}", "-t", f"{finish - origin + 0.1:.3f}", "-i", info.path]
            local += [(a - origin, b - origin, sp) for a, b, sp in run]
            # Put both streams on one clock starting at the seek point before cutting: constant frame
            # rate video and gap-free audio. Variable frame rate recordings (OBS, phones) and audio
            # that starts later than the video (Twitch VODs, MKV/TS) drift out of sync otherwise.
            graph.append(f"[{index}:v:0]fps={fps_expression(info.fps)}:start_time=0[vn{index}]")
            if info.has_audio:
                graph.append(f"{audio_input(index, self.audio_track, info.track_count)}"
                             f"aresample=async=1:first_pts=0[an{index}]")
            if len(run) == 1:
                vsrc.append(f"[vn{index}]")
                asrc.append(f"[an{index}]")
                continue
            vlabels = [f"[vs{index}_{j}]" for j in range(len(run))]
            alabels = [f"[as{index}_{j}]" for j in range(len(run))]
            graph.append(f"[vn{index}]split={len(run)}{''.join(vlabels)}")
            if info.has_audio:
                graph.append(f"[an{index}]asplit={len(run)}{''.join(alabels)}")
            vsrc += vlabels
            asrc += alabels
        silent_input = len(runs)
        if not info.has_audio:
            args += ["-f", "lavfi", "-t", f"{out_duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        overlay_input = silent_input + (0 if info.has_audio else 1)
        if overlay is not None and layout:
            args += overlay.input_args()

        count = len(local)
        if count == 1:
            a, b, sp = local[0]
            vpts = "PTS-STARTPTS" if sp == 1.0 else f"(PTS-STARTPTS)/{sp:g}"
            graph.append(f"{vsrc[0]}trim=start={a:.3f}:end={b:.3f},setpts={vpts}[vcat]")
            if info.has_audio:
                tempo = "" if sp == 1.0 else f",atempo={sp:g}"
                graph.append(f"{asrc[0]}atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS{tempo}[acat]")
        else:
            for i, (a, b, sp) in enumerate(local):
                vpts = "PTS-STARTPTS" if sp == 1.0 else f"(PTS-STARTPTS)/{sp:g}"
                graph.append(f"{vsrc[i]}trim=start={a:.3f}:end={b:.3f},setpts={vpts}[v{i}]")
                if info.has_audio:
                    tempo = "" if sp == 1.0 else f",atempo={sp:g}"
                    graph.append(
                        f"{asrc[i]}atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS{tempo},"
                        f"afade=t=in:d={MICRO_FADE},afade=t=out:st={max(0.0, (b - a) / sp - MICRO_FADE):.3f}:d={MICRO_FADE}[a{i}]"
                    )
            if info.has_audio:
                pairs = "".join(f"[v{i}][a{i}]" for i in range(count))
                graph.append(f"{pairs}concat=n={count}:v=1:a=1[vcat][acat]")
            else:
                graph.append("".join(f"[v{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[vcat]")

        # Video post-processing.
        current = "[vcat]"
        video_steps = [f"fps={fps_expression(info.fps)}"]
        if info.width % 2 or info.height % 2:
            video_steps.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        if speed != 1.0:
            video_steps += [f"setpts=PTS/{speed:g}", f"fps={fps_expression(info.fps)}"]
            out_duration /= speed
        graph.append(f"{current}{','.join(video_steps)}[vfps]")
        current = "[vfps]"
        if layout:
            graph.append(vertical_filter(current, "[vlay]", info, layout, subject_x, facecam, self.blur,
                                         game_zoom=game_zoom, game_shift=game_shift, split_top=split_top,
                                         game_area=game_area))
            current = "[vlay]"
            if overlay is not None:
                graph.append(overlay.filter(current, overlay_input, "[vovr]"))
                current = "[vovr]"
        tail = []
        if fade > 0:
            tail.append(f"fade=t=in:st=0:d={fade:.3f},fade=t=out:st={max(0.0, out_duration - fade):.3f}:d={fade:.3f}")
        if video_effects:
            tail.append(video_effects)
        if captions_file:
            tail.append(f"ass={captions_file}")
        tail.append("format=yuv420p")
        graph.append(f"{current}{','.join(tail)}[vout]")

        # Audio post-processing.
        edge_fade = max(fade, MICRO_FADE)
        audio_steps = [f"atempo={speed:g}"] if speed != 1.0 else []
        audio_steps += [f"afade=t=in:d={edge_fade:.3f}",
                       f"afade=t=out:st={max(0.0, out_duration - edge_fade):.3f}:d={edge_fade:.3f}"]
        if audio_filter:
            audio_steps.append(audio_filter)
        audio_steps.append("aresample=48000")
        audio_src = "[acat]" if info.has_audio else f"[{silent_input}:a]"
        graph.append(f"{audio_src}{','.join(audio_steps)}[aout]")

        args += [
            "-filter_complex", ";".join(graph),
            "-map", "[vout]", "-map", "[aout]",
            *video_codec_args(self.encoder, self.quality),
            *self.audio_args,
            "-t", f"{out_duration:.3f}",
            "-movflags", "+faststart",
            *extra_output_args,
            output,
        ]
        return args

    def render_segment(self, ranges: Sequence[Sequence[float]], output: Path,
                       on_progress: Optional[Callable[[float], None]] = None,
                       cwd: Optional[str] = None, **options) -> None:
        output = Path(output)
        tmp = output.with_name(f".{output.stem}.partial{output.suffix}")
        args = self.build_segment_command(ranges, str(tmp.resolve()), **options)
        try:
            run_ffmpeg(args, duration=pieces_duration(ranges), on_progress=on_progress,
                       cancel_event=self.cancel_event, cwd=cwd)
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- loudness ----------------------------------------------------------------------------------
    def measure_segment_loudness(self, ranges: Sequence[Sequence[float]], target_lufs: float) -> Optional[dict]:
        if not self.info.has_audio or not ranges:
            return None
        origin, finish = min(r[0] for r in ranges), max(r[1] for r in ranges)
        source = audio_input(0, self.audio_track, self.info.track_count)
        args = ["-ss", f"{origin:.3f}", "-t", f"{finish - origin:.3f}", "-i", self.info.path,
                "-filter_complex", f"{source}{loudnorm_measure_filter(target_lufs)}[m]", "-map", "[m]",
                "-f", "null", "-"]
        stderr = run_ffmpeg(args, cancel_event=self.cancel_event, loglevel="info")
        return parse_loudnorm(stderr)

    # -- concatenation -------------------------------------------------------------------------------
    def concat(self, files: Sequence[Path], output: Path, work_dir: Path, total: float,
               normalize: bool, target_lufs: float,
               on_progress: Optional[Callable[[float], None]] = None) -> None:
        list_file = work_dir / "concat.txt"
        with list_file.open("w", encoding="utf-8") as handle:
            for f in files:
                escaped = str(Path(f).resolve()).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        source = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
        tmp = output.with_name(f".{output.stem}.partial{output.suffix}")
        try:
            if normalize:
                measure = run_ffmpeg(
                    [*source, "-vn", "-af", loudnorm_measure_filter(target_lufs), "-f", "null", "-"],
                    duration=total, cancel_event=self.cancel_event, loglevel="info",
                    on_progress=(lambda p: on_progress(p * 0.4)) if on_progress else None,
                )
                apply = loudnorm_apply_filter(parse_loudnorm(measure), target_lufs)
                args = [*source, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
                        "-af", f"{apply},aresample=48000", *self.audio_args,
                        "-movflags", "+faststart", str(tmp)]
                progress = (lambda p: on_progress(0.4 + p * 0.6)) if on_progress else None
            else:
                args = [*source, "-c", "copy", "-movflags", "+faststart", str(tmp)]
                progress = on_progress
            run_ffmpeg(args, duration=total, on_progress=progress, cancel_event=self.cancel_event)
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()
            if list_file.exists():
                list_file.unlink()
