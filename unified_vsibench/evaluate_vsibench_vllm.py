from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .score_vsibench import MCA_QUESTION_TYPES, NA_QUESTION_TYPES, aggregate, score_record, write_json


DEFAULT_PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a numerical value only."


@dataclass
class EvalConfig:
    model: str
    output_dir: str
    split: str
    dataset_path: str
    hf_cache_dir: str | None
    data_file: str | None
    video_root: str | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    batch_size: int
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    limit: int | None
    num_frames: int
    fps: float
    trust_remote_code: bool
    prompt_style: str


def read_table(path: Path) -> Any:
    import pandas as pd
    from datasets import Dataset

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file extension: {path}")
    return Dataset.from_pandas(df, preserve_index=False)


def load_vsibench(
    dataset_path: str,
    split: str,
    data_file: str | None,
    limit: int | None,
    hf_cache_dir: str | None,
) -> list[dict[str, Any]]:
    if data_file:
        dataset = read_table(Path(data_file))
    else:
        from datasets import load_dataset

        dataset = load_dataset(dataset_path, split=split, video=True, cache_dir=hf_cache_dir)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def normalize_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, list):
        return [str(option) for option in options]
    if hasattr(options, "tolist"):
        return [str(option) for option in options.tolist()]
    if isinstance(options, str):
        stripped = options.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, list):
                return [str(option) for option in decoded]
        except json.JSONDecodeError:
            pass
        return [line.strip() for line in stripped.splitlines() if line.strip()]
    return [str(options)]


def build_prompt(row: dict[str, Any], prompt_style: str) -> str:
    question = str(row["question"])
    question_type = row["question_type"]
    parts = [DEFAULT_PRE_PROMPT, question]

    options = normalize_options(row.get("options"))
    if question_type in MCA_QUESTION_TYPES:
        if options:
            parts.append("Options:\n" + "\n".join(options))
        parts.append(MCA_POST_PROMPT)
    elif question_type in NA_QUESTION_TYPES:
        parts.append(NA_POST_PROMPT)
    else:
        raise ValueError(f"Unknown VSI-Bench question_type: {question_type}")

    if prompt_style == "thinking":
        parts.append("Think briefly if needed, then put the final answer in <answer></answer> tags.")
    return "\n".join(parts)


def resolve_video_path(row: dict[str, Any], video_root: str | None, hf_cache_dir: str | None) -> str:
    candidates: list[str] = []
    for key in ("video_path", "video", "path"):
        value = row.get(key)
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, dict):
            for subkey in ("path", "filename"):
                subvalue = value.get(subkey)
                if isinstance(subvalue, str):
                    candidates.append(subvalue)

    dataset = row.get("dataset")
    scene = row.get("scene_name")
    if dataset and scene:
        relative_scene_path = str(Path(str(dataset)) / f"{scene}.mp4")
        candidates.append(relative_scene_path)
        cache_roots = []
        if hf_cache_dir:
            cache_roots.append(Path(hf_cache_dir).expanduser())
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            cache_roots.append(Path(hf_home).expanduser() / "vsibench")
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "vsibench")
        for cache_root in cache_roots:
            candidates.append(str(cache_root / relative_scene_path))

    if video_root:
        rooted = [str(Path(video_root) / candidate) for candidate in candidates]
        candidates = rooted + candidates

    for candidate in candidates:
        if Path(candidate).exists():
            return str(Path(candidate))

    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot resolve video path for sample: {row}")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_vllm(model: str, args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1, "image": 0},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    return llm, sampling


def build_vllm_inputs(rows: list[dict[str, Any]], processor: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    from qwen_vl_utils import process_vision_info

    requests: list[dict[str, Any]] = []
    for row in rows:
        prompt_text = build_prompt(row, args.prompt_style)
        video_path = resolve_video_path(row, args.video_root, args.hf_cache_dir)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": args.fps,
                        "nframes": args.num_frames,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data: dict[str, Any] = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "metadata": {"prompt_text": prompt_text, "video_path": video_path},
            }
        )
    return requests


def evaluate(args: argparse.Namespace) -> tuple[Path, Path]:
    from transformers import AutoProcessor
    from tqdm import tqdm

    rows = load_vsibench(args.dataset_path, args.split, args.data_file, args.limit, args.hf_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{args.run_name}.predictions.jsonl"
    metrics_path = output_dir / f"{args.run_name}.metrics.json"
    config_path = output_dir / f"{args.run_name}.config.json"

    config = EvalConfig(
        model=args.model,
        output_dir=args.output_dir,
        split=args.split,
        dataset_path=args.dataset_path,
        hf_cache_dir=args.hf_cache_dir,
        data_file=args.data_file,
        video_root=args.video_root,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        num_frames=args.num_frames,
        fps=args.fps,
        trust_remote_code=args.trust_remote_code,
        prompt_style=args.prompt_style,
    )
    write_json(config_path, asdict(config))

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"

    llm, sampling = load_vllm(args.model, args)
    scored_records: list[dict[str, Any]] = []

    with prediction_path.open("w", encoding="utf-8") as f:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="VSI-Bench vLLM inference"):
            requests = build_vllm_inputs(batch_rows, processor, args)
            llm_requests = [
                {key: value for key, value in request.items() if key != "metadata"}
                for request in requests
            ]
            outputs = llm.generate(llm_requests, sampling_params=sampling)
            for row, request, output in zip(batch_rows, requests, outputs):
                prediction = output.outputs[0].text.strip() if output.outputs else ""
                record = {
                    "sample_id": row.get("id", row.get("question_id", row.get("index"))),
                    "dataset": row.get("dataset"),
                    "scene_name": row.get("scene_name"),
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "options": normalize_options(row.get("options")),
                    "ground_truth": row["ground_truth"],
                    "prediction": prediction,
                    "prompt": request["metadata"]["prompt_text"],
                    "video_path": request["metadata"]["video_path"],
                    "model": args.model,
                }
                scored = score_record(record)
                scored_records.append(scored)
                f.write(json.dumps(scored, ensure_ascii=False) + "\n")
                f.flush()

    metrics = aggregate(scored_records)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return prediction_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified vLLM evaluator for VSI-Bench.")
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dataset-path", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--hf-cache-dir", default=os.getenv("VSI_HF_CACHE_DIR", "vsibench"))
    parser.add_argument("--data-file", default=None, help="Optional local parquet/json/jsonl/csv export of VSI-Bench.")
    parser.add_argument("--video-root", default=None, help="Optional root containing <dataset>/<scene_name>.mp4 files.")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TP_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--prompt-style", choices=["default", "thinking"], default="default")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = args.model.rstrip("/").replace("/", "__")
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
