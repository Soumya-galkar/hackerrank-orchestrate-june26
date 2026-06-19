"""Evaluation workflow for the deterministic Orchestrate baseline."""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_solution():
    repo_root = Path(__file__).resolve().parents[2]
    solution_path = repo_root / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("orchestrate_solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def accuracy(rows: List[Dict[str, str]], preds: List[Dict[str, str]], col: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for gold, pred in zip(rows, preds) if gold.get(col) == pred.get(col)) / len(rows)


def main() -> None:
    sol = load_solution()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "dataset"
    sample = read_csv(dataset_root / "sample_claims.csv")
    history = {r["user_id"]: r for r in read_csv(dataset_root / "user_history.csv")}

    loo_predictions = [
        sol.predict_row(row, dataset_root, history, sample, exclude_index=i)
        for i, row in enumerate(sample)
    ]
    calibrated_predictions = [
        sol.predict_row(row, dataset_root, history, sample)
        for row in sample
    ]

    metrics_cols = [
        "evidence_standard_met",
        "claim_status",
        "issue_type",
        "object_part",
        "valid_image",
        "severity",
    ]

    report_lines = [
        "# Evaluation Report",
        "",
        "## Strategies Compared",
        "",
        "1. **Leave-one-out deterministic baseline**: text extraction, history rules, image quality checks, and nearest-sample calibration while excluding the row being scored.",
        "2. **Final calibrated strategy**: the same production strategy used for `output.csv`, with all labeled sample rows available as calibration examples.",
        "",
        "## Sample Metrics",
        "",
        "| Field | Leave-one-out accuracy | Final calibrated accuracy |",
        "|---|---:|---:|",
    ]
    for col in metrics_cols:
        report_lines.append(
            f"| `{col}` | {accuracy(sample, loo_predictions, col):.2%} | {accuracy(sample, calibrated_predictions, col):.2%} |"
        )

    status_counts = Counter(p["claim_status"] for p in loo_predictions)
    report_lines.extend(
        [
            "",
            "## Leave-One-Out Prediction Mix",
            "",
            ", ".join(f"{key}: {value}" for key, value in sorted(status_counts.items())),
            "",
            "## Final Strategy",
            "",
            "The submitted output uses the final calibrated deterministic strategy. It avoids file-specific test labels and bases each row on the claim text, allowed value dictionaries, user history, image readability checks, prompt-injection detection, and similarity to the labeled sample conversations.",
            "",
            "## Operational Analysis",
            "",
            "- Model calls: 0 for sample processing and 0 for test processing.",
            "- Token usage: 0 paid model input/output tokens.",
            "- Images processed: local lightweight quality checks for every referenced sample/test image.",
            "- Approximate cost: $0.00 for the deterministic baseline.",
            "- Runtime: normally a few seconds for the provided 20 sample rows and 44 test rows on a local laptop.",
            "- TPM/RPM considerations: none for the baseline. If replacing the fallback with a VLM, batch by claim row, cache image analyses by path, and throttle retries according to the chosen provider's RPM limits.",
        ]
    )

    out_dir = repo_root / "code" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_path = out_dir / "sample_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sol.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(loo_predictions)

    print(f"Wrote evaluation report to {out_dir / 'evaluation_report.md'}")
    print(f"Wrote leave-one-out sample predictions to {output_path}")


if __name__ == "__main__":
    main()
