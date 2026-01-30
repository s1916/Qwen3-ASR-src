#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class TimestampToken:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    text: str


def run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                " ".join(shlex.quote(c) for c in cmd), proc.stdout, proc.stderr
            )
        )


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            run([tool, "-version"])
        except Exception as exc:
            raise RuntimeError(f"Missing {tool}. Please install ffmpeg.") from exc


def extract_audio(input_path: str, wav_path: str) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            wav_path,
        ]
    )


def split_audio(wav_path: str, out_dir: str, segment_seconds: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "seg_%04d.wav")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            pattern,
        ]
    )
    segments = sorted(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")]
    )
    if not segments:
        raise RuntimeError("No audio segments were created.")
    return segments


def get_duration(path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return float(proc.stdout.strip())


def load_model(backend: str, model_name: str, dtype: str, device: str, max_new_tokens: int, max_batch: int, use_forced_aligner: bool, forced_aligner: Optional[str]):
    import torch
    from qwen_asr import Qwen3ASRModel

    torch_dtype = getattr(torch, dtype)
    if backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model_name,
            dtype=torch_dtype,
            tensor_parallel_size=1,
            max_num_seqs=max_batch,
        )
        return model

    if use_forced_aligner and forced_aligner:
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device,
            max_inference_batch_size=max_batch,
            max_new_tokens=max_new_tokens,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=dict(
                dtype=torch_dtype,
                device_map=device,
            ),
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device,
            max_inference_batch_size=max_batch,
            max_new_tokens=max_new_tokens,
        )
    return model


def normalize_timestamp_token(item) -> Optional[TimestampToken]:
    if item is None:
        return None

    if isinstance(item, dict):
        start = item.get("start")
        end = item.get("end")
        text = item.get("text", "")
        if start is None or end is None:
            return None
        return TimestampToken(float(start), float(end), str(text))

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        start = float(item[0])
        end = float(item[1])
        text = ""
        if len(item) >= 3:
            text = str(item[2])
        return TimestampToken(start, end, text)

    return None


def serialize_time_stamps(items) -> List[dict]:
    serialized: List[dict] = []
    if not items:
        return serialized
    for item in items:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = item[0]
            end = item[1]
            text = item[2] if len(item) >= 3 else ""
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            text = getattr(item, "text", "")
        if start is None or end is None:
            continue
        serialized.append({"start": float(start), "end": float(end), "text": str(text)})
    return serialized


def smart_append(buffer: str, piece: str) -> str:
    if not buffer:
        return piece
    if not piece:
        return buffer
    if piece[:1].isspace() or buffer[-1:].isspace():
        return buffer + piece
    if piece[0] in ")]}%.,!?;:" or buffer[-1] in "([{":
        return buffer + piece
    if buffer[-1].isalnum() and piece[0].isalnum():
        return buffer + " " + piece
    return buffer + piece


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_cues_from_tokens(tokens: List[TimestampToken], offset: float, max_chars: int, max_duration: float) -> List[Cue]:
    cues: List[Cue] = []
    current_text = ""
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def should_break(last_piece: str, total_text: str, start: float, end: float) -> bool:
        if re.search(r"[.!?。！？]$", last_piece.strip()):
            return True
        if len(total_text) >= max_chars and (end - start) >= 1.0:
            return True
        if (end - start) >= max_duration:
            return True
        return False

    for token in tokens:
        if not token.text:
            continue
        if current_start is None:
            current_start = token.start
        current_end = token.end
        current_text = smart_append(current_text, token.text)
        if should_break(token.text, current_text, current_start, current_end):
            cues.append(Cue(current_start + offset, current_end + offset, clean_text(current_text)))
            current_text = ""
            current_start = None
            current_end = None

    if current_text and current_start is not None and current_end is not None:
        cues.append(Cue(current_start + offset, current_end + offset, clean_text(current_text)))

    return cues


def fallback_cues_from_text(text: str, offset: float, duration: float, max_chars: int) -> List[Cue]:
    parts = re.split(r"([.!?。！？])", text)
    sentences = []
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        if re.search(r"[.!?。！？]$", part):
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)

    lines: List[str] = []
    for sent in sentences:
        sent = clean_text(sent)
        if not sent:
            continue
        while len(sent) > max_chars:
            lines.append(sent[:max_chars])
            sent = sent[max_chars:]
        lines.append(sent)

    if not lines:
        return []

    per = max(duration / len(lines), 0.5)
    cues: List[Cue] = []
    t = 0.0
    for line in lines:
        cues.append(Cue(offset + t, offset + min(t + per, duration), line))
        t += per
    return cues


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: List[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n")
            f.write(cue.text + "\n\n")

def save_progress(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_progress(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transcribe_segments(model, segment_paths: List[str], language: Optional[str], return_time_stamps: bool):
    if language is None:
        lang_arg = None
    else:
        lang_arg = [language] * len(segment_paths)

    return model.transcribe(
        audio=segment_paths,
        language=lang_arg,
        return_time_stamps=return_time_stamps,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the movie file or audio file")
    parser.add_argument("--output", required=True, help="Output SRT path")
    parser.add_argument("--tmp", default="./tmp_asr", help="Temporary directory")
    parser.add_argument("--segment-seconds", type=int, default=300, help="Segment length in seconds")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default=None, help="Language name or None for auto detection")
    parser.add_argument("--forced-aligner", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Forced aligner model name or local path")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=6.0)
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument("--retries", type=int, default=2, help="Retry count per batch on failure")
    parser.add_argument("--retry-backoff", type=float, default=5.0, help="Seconds to wait between retries")
    args = parser.parse_args()

    check_ffmpeg()

    os.makedirs(args.tmp, exist_ok=True)
    wav_path = os.path.join(args.tmp, "audio_16k.wav")
    extract_audio(args.input, wav_path)

    segments_dir = os.path.join(args.tmp, "segments")
    if os.path.exists(segments_dir):
        for f in os.listdir(segments_dir):
            os.remove(os.path.join(segments_dir, f))
    segments = split_audio(wav_path, segments_dir, args.segment_seconds)
    durations = [get_duration(p) for p in segments]

    model = load_model(
        backend=args.backend,
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_batch=args.max_batch,
        use_forced_aligner=not args.no_timestamps,
        forced_aligner=args.forced_aligner if not args.no_timestamps else None,
    )

    progress_path = os.path.join(args.tmp, "progress.json")
    progress = None
    if args.resume:
        progress = load_progress(progress_path)
    if progress is None:
        progress = {
            "input": os.path.abspath(args.input),
            "segments": [os.path.basename(p) for p in segments],
            "segment_seconds": args.segment_seconds,
            "no_timestamps": args.no_timestamps,
            "completed": {},
        }
        save_progress(progress_path, progress)
    else:
        if progress.get("input") != os.path.abspath(args.input):
            raise RuntimeError("Resume input mismatch. Use a fresh --tmp directory.")
        if progress.get("segments") != [os.path.basename(p) for p in segments]:
            raise RuntimeError("Resume segments mismatch. Use a fresh --tmp directory.")

    cues: List[Cue] = []
    offset = 0.0
    batch_size = args.max_batch
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(segments))))
        pending = [idx for idx in batch_indices if str(idx) not in progress["completed"]]
        if not pending:
            for j, seg_path in enumerate(batch):
                seg_duration = durations[i + j]
                offset += seg_duration
            continue

        pending_paths = [segments[idx] for idx in pending]
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                results = transcribe_segments(
                    model,
                    pending_paths,
                    language=args.language,
                    return_time_stamps=not args.no_timestamps,
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt < args.retries:
                    time.sleep(args.retry_backoff)
        if last_err is not None:
            raise last_err

        for idx, result in zip(pending, results):
            entry = {"text": getattr(result, "text", "")}
            if not args.no_timestamps and getattr(result, "time_stamps", None):
                entry["time_stamps"] = serialize_time_stamps(result.time_stamps)
            progress["completed"][str(idx)] = entry
        save_progress(progress_path, progress)

        for j, seg_path in enumerate(batch):
            seg_duration = durations[i + j]
            cached = progress["completed"].get(str(i + j))
            if cached:
                if not args.no_timestamps and cached.get("time_stamps"):
                    tokens = []
                    for item in cached["time_stamps"]:
                        tok = normalize_timestamp_token(item)
                        if tok:
                            tokens.append(tok)
                    cues.extend(
                        build_cues_from_tokens(tokens, offset, args.max_chars, args.max_duration)
                    )
                else:
                    cues.extend(
                        fallback_cues_from_text(
                            cached.get("text", ""), offset, seg_duration, args.max_chars
                        )
                    )
            offset += seg_duration

    write_srt(cues, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
