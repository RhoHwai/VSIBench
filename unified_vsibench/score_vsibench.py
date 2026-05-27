from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}

NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def extract_tagged_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def fuzzy_choice(prediction: str) -> str:
    pred = extract_tagged_answer(prediction).strip()
    match = re.search(r"\b([A-Z])\b", pred.upper())
    if match:
        return match.group(1)
    match = re.search(r"^([A-Z])[\).\s:]", pred.upper())
    return match.group(1) if match else pred.split(" ")[0].rstrip(".").strip().upper()


def fuzzy_number(prediction: str) -> float | None:
    pred = extract_tagged_answer(prediction).strip().lower()
    number_words = {
        "zero": "0",
        "one": "1",
        "a": "1",
        "an": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
    }
    for word, digit in number_words.items():
        if re.search(rf"\b{word}\b", pred):
            return float(digit)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", pred)
    return float(match.group(0)) if match else None


def mean_relative_accuracy(pred: float | None, target: float | None) -> float:
    if pred is None or target is None or target == 0:
        return 0.0
    thresholds = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))
    rel_error = abs(pred - target) / abs(target)
    return float((rel_error <= 1 - thresholds).mean())


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    question_type = record["question_type"]
    target = str(record.get("ground_truth", "")).strip()
    prediction = str(record.get("prediction", record.get("predicted_answer", "")))

    if question_type in MCA_QUESTION_TYPES:
        score = 1.0 if fuzzy_choice(prediction).lower() == target.lower() else 0.0
        record["metric"] = "accuracy"
        record["score"] = score
        record["accuracy"] = score
    elif question_type in NA_QUESTION_TYPES:
        pred_value = fuzzy_number(prediction)
        try:
            target_value = float(target)
        except ValueError:
            target_value = None
        score = mean_relative_accuracy(pred_value, target_value)
        record["metric"] = "MRA:.5:.95:.05"
        record["score"] = score
        record["MRA:.5:.95:.05"] = score
        record["parsed_prediction"] = pred_value
    else:
        raise ValueError(f"Unknown VSI-Bench question_type: {question_type}")

    return record


def aggregate(records: list[dict[str, Any]]) -> OrderedDict[str, float | str]:
    if not records:
        raise ValueError("No prediction records were provided.")

    df = pd.DataFrame(records)
    output: dict[str, float] = {}

    for question_type, group in df.groupby("question_type"):
        if question_type in MCA_QUESTION_TYPES:
            output[f"{question_type}_accuracy"] = float(group["accuracy"].mean())
        elif question_type in NA_QUESTION_TYPES:
            output[f"{question_type}_MRA:.5:.95:.05"] = float(group["MRA:.5:.95:.05"].mean())

    direction_keys = [
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    ]
    if all(key in output for key in direction_keys):
        output["object_rel_direction_accuracy"] = float(np.mean([output.pop(key) for key in direction_keys]))

    ordered_keys = [
        "object_counting_MRA:.5:.95:.05",
        "object_abs_distance_MRA:.5:.95:.05",
        "object_size_estimation_MRA:.5:.95:.05",
        "room_size_estimation_MRA:.5:.95:.05",
        "object_rel_distance_accuracy",
        "object_rel_direction_accuracy",
        "route_planning_accuracy",
        "obj_appearance_order_accuracy",
    ]
    available = [key for key in ordered_keys if key in output]
    overall = float(np.mean([output[key] for key in available])) if available else float(df["score"].mean())

    results: OrderedDict[str, float | str] = OrderedDict()
    results["overall"] = overall * 100.0
    for key in ordered_keys:
        if key in output:
            results[key] = output[key] * 100.0
    results["num_samples"] = float(len(records))
    results["tabulated_keys"] = ", ".join([key for key in results if key != "tabulated_keys"])
    results["tabulated_results"] = ", ".join(
        f"{value:.3f}" for key, value in results.items() if key != "tabulated_keys" and isinstance(value, float)
    )
    return results


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VSI-Bench prediction JSONL files.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scored-output", type=Path)
    args = parser.parse_args()

    records = [score_record(record) for record in load_jsonl(args.predictions)]
    metrics = aggregate(records)

    output = args.output or args.predictions.with_suffix(".metrics.json")
    write_json(output, metrics)
    if args.scored_output:
        args.scored_output.parent.mkdir(parents=True, exist_ok=True)
        with args.scored_output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
