from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_hwaccel_lock = threading.Lock()
_hwaccel_cuda_cached: bool | None = None


def detect_cuda_hwaccel_available() -> bool:
    global _hwaccel_cuda_cached
    if _hwaccel_cuda_cached is not None:
        return _hwaccel_cuda_cached
    with _hwaccel_lock:
        if _hwaccel_cuda_cached is not None:
            return _hwaccel_cuda_cached
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            out = (r.stdout or "") + (r.stderr or "")
            _hwaccel_cuda_cached = "cuda" in out.lower()
        except Exception:
            _hwaccel_cuda_cached = False
        return _hwaccel_cuda_cached


def nvdec_input_prefix(*, use_nvdec: bool) -> list[str]:
    if not use_nvdec:
        return []
    if not detect_cuda_hwaccel_available():
        logger.warning(
            "ENABLE_NVDEC requested but ffmpeg cuda hwaccel not listed; using software decode"
        )
        return []
    return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]


def rtsp_input_options() -> list[str]:
    # Socket I/O timeout in microseconds (replaces deprecated -stimeout in current FFmpeg).
    return [
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
    ]


def _run_ffmpeg(args: list[str], *, cwd: Path | None = None) -> None:
    logger.debug("ffmpeg %s", " ".join(args))
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()[-4000:]
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {msg}")


def _run_ffmpeg_attempts(candidates: list[list[str]], *, cwd: Path | None = None) -> None:
    last: Exception | None = None
    for args in candidates:
        try:
            _run_ffmpeg(args, cwd=cwd)
            return
        except RuntimeError as e:
            last = e
    assert last is not None
    raise last


def probe_duration_seconds(path: str) -> float | None:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if r.returncode != 0:
            return None
        v = (r.stdout or "").strip()
        return float(v) if v else None
    except Exception:
        return None


def build_input_args(uri: str, kind: str, *, use_nvdec: bool) -> list[str]:
    opts: list[str] = []
    opts.extend(nvdec_input_prefix(use_nvdec=use_nvdec))
    if kind == "rtsp":
        opts.extend(rtsp_input_options())
    opts.extend(["-i", uri])
    return opts


def _jpg_vf_fps(chunk_seconds: float, frames_per_chunk: int, *, use_nvdec: bool) -> str:
    fps = frames_per_chunk / max(chunk_seconds, 0.5)
    if use_nvdec and detect_cuda_hwaccel_available():
        return f"hwdownload,format=nv12,fps={fps}"
    return f"fps={fps}"


def segment_to_jpg(
    *,
    uri: str,
    kind: str,
    out_dir: Path,
    chunk_seconds: float,
    use_nvdec: bool,
    frames_per_chunk: int = 1,
    max_chunks: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%06d.jpg")
    vf = _jpg_vf_fps(chunk_seconds, frames_per_chunk, use_nvdec=use_nvdec)
    frame_cap: list[str] = []
    if max_chunks is not None:
        frame_cap = ["-frames:v", str(max_chunks * frames_per_chunk)]
    attempts: list[list[str]] = []
    if use_nvdec and nvdec_input_prefix(use_nvdec=True):
        inp = build_input_args(uri, kind, use_nvdec=True)
        attempts.append(
            [
                *inp,
                "-an",
                "-vf",
                vf,
                *frame_cap,
                "-q:v",
                "3",
                pattern,
            ]
        )
    attempts.append(
        [
            *build_input_args(uri, kind, use_nvdec=False),
            "-an",
            "-vf",
            f"fps={frames_per_chunk / max(chunk_seconds, 0.5)}",
            *frame_cap,
            "-q:v",
            "3",
            pattern,
        ]
    )
    _run_ffmpeg_attempts(attempts)
    return sorted(out_dir.glob("chunk_*.jpg"))


def extract_spaced_jpegs_from_mp4(
    mp4_path: Path,
    *,
    out_dir: Path,
    stem: str,
    n: int,
    use_nvdec: bool,
) -> list[Path]:
    """Extract n JPEGs evenly spaced over the file's timeline (same rule as fps=n/duration)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{stem}_%03d.jpg")
    duration = probe_duration_seconds(str(mp4_path)) or 1.0
    fps = n / max(duration, 0.01)
    vf_soft = f"fps={fps}"
    attempts: list[list[str]] = []
    if use_nvdec and nvdec_input_prefix(use_nvdec=True):
        vf_hw = f"hwdownload,format=nv12,fps={fps}"
        attempts.append(
            [
                *nvdec_input_prefix(use_nvdec=True),
                "-i",
                str(mp4_path),
                "-an",
                "-vf",
                vf_hw,
                "-frames:v",
                str(n),
                "-q:v",
                "3",
                pattern,
            ]
        )
    attempts.append(
        [
            "-i",
            str(mp4_path),
            "-an",
            "-vf",
            vf_soft,
            "-frames:v",
            str(n),
            "-q:v",
            "3",
            pattern,
        ]
    )
    _run_ffmpeg_attempts(attempts)
    paths = sorted(out_dir.glob(f"{stem}_*.jpg"))
    if len(paths) < n:
        raise RuntimeError(
            f"expected {n} jpeg(s) from {mp4_path.name}, got {len(paths)}"
        )
    return paths[:n]


def segment_to_mp4(
    *,
    uri: str,
    kind: str,
    out_dir: Path,
    chunk_seconds: float,
    use_nvdec: bool,
    max_chunks: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%06d.mp4")
    seg_args = [
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
    ]
    time_cap: list[str] = []
    if max_chunks is not None:
        dur = max(float(max_chunks) * chunk_seconds, chunk_seconds)
        time_cap = ["-t", str(dur)]

    attempts: list[list[str]] = []
    attempts.append(
        [
            *build_input_args(uri, kind, use_nvdec=False),
            *time_cap,
            *seg_args,
            "-c",
            "copy",
            pattern,
        ]
    )
    if use_nvdec and nvdec_input_prefix(use_nvdec=True):
        attempts.append(
            [
                *build_input_args(uri, kind, use_nvdec=True),
                *time_cap,
                *seg_args,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                pattern,
            ]
        )
    attempts.append(
        [
            *build_input_args(uri, kind, use_nvdec=False),
            *time_cap,
            *seg_args,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            pattern,
        ]
    )
    _run_ffmpeg_attempts(attempts)
    return sorted(out_dir.glob("chunk_*.mp4"))


def extract_representative_jpeg(
    mp4_path: Path,
    out_jpg: Path,
    *,
    use_nvdec: bool,
) -> None:
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = f"{out_jpg.stem}_one"
    paths = extract_spaced_jpegs_from_mp4(
        mp4_path,
        out_dir=out_jpg.parent,
        stem=tmp_stem,
        n=1,
        use_nvdec=use_nvdec,
    )
    shutil.move(str(paths[0]), str(out_jpg))
