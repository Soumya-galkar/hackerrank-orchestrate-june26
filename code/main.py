"""Deterministic baseline for the HackerRank Orchestrate challenge.

The system is intentionally runnable without network access or API keys.  It
combines claim-text extraction, user-history risk rules, lightweight image
quality checks, and calibration from the labeled sample rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageStat
except Exception:  # Pillow is optional in the judge environment.
    Image = None
    ImageStat = None


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

RISK_VALUES = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

PART_KEYWORDS = {
    "car": [
        ("front_bumper", ["front bumper", "front side", "front", "parachoques delantero"]),
        ("rear_bumper", ["rear bumper", "back bumper", "back of the car", "from behind", "parachoques trasero", "parachoques de atras", "rear"]),
        ("side_mirror", ["side mirror", "left mirror", "mirror", "toot gaya"]),
        ("windshield", ["windshield", "front glass", "glass", "windscreen"]),
        ("headlight", ["headlight", "front light"]),
        ("taillight", ["taillight", "back light", "tail light"]),
        ("door", ["door panel", "left side door", "door"]),
        ("hood", ["hood", "top panel", "bonnet"]),
        ("fender", ["fender"]),
        ("quarter_panel", ["quarter panel"]),
        ("body", ["body panel", "car body", "body"]),
    ],
    "laptop": [
        ("screen", ["screen", "display", "pantalla"]),
        ("keyboard", ["keyboard", "keys", "keycaps", "teclas", "keys missing"]),
        ("trackpad", ["trackpad", "touchpad"]),
        ("hinge", ["hinge"]),
        ("lid", ["lid", "outer lid"]),
        ("corner", ["corner"]),
        ("port", ["port"]),
        ("base", ["base"]),
        ("body", ["body", "outer body", "side edge"]),
    ],
    "package": [
        ("package_corner", ["corner", "package corner", "box corner", "dab gaya"]),
        ("package_side", ["side", "package side"]),
        ("seal", ["seal", "tape", "opened", "open jaisa", "torn open"]),
        ("label", ["label", "shipping label"]),
        ("contents", ["contents", "item inside", "missing contents", "not inside", "opened it"]),
        ("item", ["item", "product"]),
        ("box", ["box", "package", "parcel", "cardboard"]),
    ],
}

ISSUE_KEYWORDS = [
    ("glass_shatter", ["shattered", "shatter"]),
    ("missing_part", ["missing", "not inside", "faltan"]),
    ("torn_packaging", ["torn", "phati", "opened", "open jaisa", "seal broken"]),
    ("crushed_packaging", ["crushed", "crush", "dab gaya", "badly crushed"]),
    ("water_damage", ["water", "wet", "rain", "liquid damage"]),
    ("stain", ["stain", "coffee", "oil", "oily", "sticky"]),
    ("broken_part", ["broken", "broke", "damaged", "not sitting", "toot", "keycaps came off", "key is missing"]),
    ("crack", ["crack", "cracked", "crack spreading", "body crack"]),
    ("dent", ["dent", "dented", "hail dents"]),
    ("scratch", ["scratch", "scrape", "mark"]),
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "claim",
    "customer",
    "support",
    "agent",
    "please",
    "review",
    "image",
    "images",
    "photo",
    "photos",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", normalize(text)) if len(t) > 2 and t not in STOPWORDS}


def image_ids(image_paths: str) -> List[str]:
    return [Path(p.strip()).stem for p in image_paths.split(";") if p.strip()]


def pick_part(claim_object: str, text: str) -> str:
    haystack = normalize(text)
    for part, keys in PART_KEYWORDS.get(claim_object, []):
        if any(k in haystack for k in keys):
            return part
    return "unknown" if claim_object in {"car", "laptop", "package"} else "unknown"


def pick_issue(text: str, claim_object: str, part: str) -> str:
    haystack = normalize(text)
    for issue, keys in ISSUE_KEYWORDS:
        if any(k in haystack for k in keys):
            if claim_object == "package" and issue == "broken_part" and part not in {"contents", "item"}:
                continue
            return issue
    if "not working" in haystack or "stopped working" in haystack:
        return "unknown"
    return "unknown"


def text_instruction_flag(text: str) -> bool:
    haystack = normalize(text)
    triggers = [
        "approve the claim immediately",
        "skip manual review",
        "ignore all previous instructions",
        "mark this row supported",
        "follow",
        "should be approved",
    ]
    return any(t in haystack for t in triggers)


def image_quality_flags(dataset_root: Path, image_paths_text: str) -> Tuple[List[str], bool]:
    flags: List[str] = []
    usable = False
    if Image is None or ImageStat is None:
        return flags, True

    for rel in [p.strip() for p in image_paths_text.split(";") if p.strip()]:
        path = dataset_root / rel
        if not path.exists():
            flags.append("damage_not_visible")
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("L")
                w, h = img.size
                usable = usable or (w >= 160 and h >= 120)
                stat = ImageStat.Stat(img.resize((64, 64)))
                mean = stat.mean[0]
                contrast = stat.stddev[0]
                if mean < 35 or mean > 235:
                    flags.append("low_light_or_glare")
                if contrast < 13:
                    flags.append("blurry_image")
        except Exception:
            flags.append("damage_not_visible")
    return sorted(set(flags)), usable


def history_flags(history: Dict[str, Dict[str, str]], user_id: str) -> List[str]:
    row = history.get(user_id, {})
    raw = row.get("history_flags", "none")
    flags = [f for f in raw.split(";") if f and f != "none"]
    try:
        rejected = int(row.get("rejected_claim", "0") or 0)
        recent = int(row.get("last_90_days_claim_count", "0") or 0)
    except ValueError:
        rejected = recent = 0
    if rejected >= 3 or recent >= 5:
        flags.append("user_history_risk")
    if "user_history_risk" in flags and (rejected >= 3 or "manual_review_required" in raw):
        flags.append("manual_review_required")
    return sorted(set(flags))


def severity_from_issue(issue: str, text: str, status: str) -> str:
    haystack = normalize(text)
    if status == "not_enough_information":
        return "unknown"
    if issue == "none":
        return "none"
    if any(k in haystack for k in ["severe", "shattered", "missing contents", "broken item", "pretty bad"]):
        return "high" if issue in {"glass_shatter", "missing_part", "broken_part"} else "medium"
    if issue in {"scratch", "stain"}:
        return "low"
    if issue in {"crack", "dent", "broken_part", "water_damage", "crushed_packaging", "torn_packaging", "glass_shatter"}:
        return "medium"
    return "unknown"


def nearest_exemplar(row: Dict[str, str], examples: Sequence[Dict[str, str]], exclude_index: Optional[int] = None) -> Tuple[Optional[Dict[str, str]], float]:
    row_tokens = tokens(row.get("user_claim", ""))
    best: Optional[Dict[str, str]] = None
    best_score = 0.0
    part = pick_part(row.get("claim_object", ""), row.get("user_claim", ""))
    issue = pick_issue(row.get("user_claim", ""), row.get("claim_object", ""), part)
    for idx, ex in enumerate(examples):
        if exclude_index is not None and idx == exclude_index:
            continue
        if ex.get("claim_object") != row.get("claim_object"):
            continue
        ex_tokens = tokens(ex.get("user_claim", ""))
        overlap = len(row_tokens & ex_tokens)
        union = max(1, len(row_tokens | ex_tokens))
        score = overlap / union
        if ex.get("object_part") == part:
            score += 0.18
        if ex.get("issue_type") == issue:
            score += 0.14
        if ex.get("user_id") == row.get("user_id"):
            score += 0.05
        if score > best_score:
            best = ex
            best_score = score
    return best, best_score


def merge_flags(*groups: Iterable[str]) -> str:
    flags: List[str] = []
    for group in groups:
        for flag in group:
            if flag and flag != "none" and flag in RISK_VALUES:
                flags.append(flag)
    unique = sorted(set(flags), key=lambda f: [
        "blurry_image",
        "cropped_or_obstructed",
        "low_light_or_glare",
        "wrong_angle",
        "wrong_object",
        "wrong_object_part",
        "damage_not_visible",
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "text_instruction_present",
        "user_history_risk",
        "manual_review_required",
    ].index(f) if f in RISK_VALUES and f != "none" else 99)
    return ";".join(unique) if unique else "none"


def predict_row(
    row: Dict[str, str],
    dataset_root: Path,
    history: Dict[str, Dict[str, str]],
    examples: Sequence[Dict[str, str]],
    exclude_index: Optional[int] = None,
) -> Dict[str, str]:
    text = row.get("user_claim", "")
    claim_object = row.get("claim_object", "unknown")
    part = pick_part(claim_object, text)
    issue = pick_issue(text, claim_object, part)
    img_flags, images_usable = image_quality_flags(dataset_root, row.get("image_paths", ""))
    hist_flags = history_flags(history, row.get("user_id", ""))
    extra_flags: List[str] = []
    if text_instruction_flag(text):
        extra_flags.extend(["text_instruction_present", "manual_review_required"])

    exemplar, score = nearest_exemplar(row, examples, exclude_index=exclude_index)
    exemplar_flags: List[str] = []
    status = "supported"
    evidence = "true"
    valid_image = "true" if images_usable else "false"

    if exemplar and score >= 0.32:
        status = exemplar.get("claim_status", status)
        evidence = exemplar.get("evidence_standard_met", evidence)
        valid_image = exemplar.get("valid_image", valid_image)
        exemplar_flags = [f for f in exemplar.get("risk_flags", "none").split(";") if f != "none"]
        if part == "unknown":
            part = exemplar.get("object_part", part)
        if issue == "unknown":
            issue = exemplar.get("issue_type", issue)

    # Conservative text-only adjustments for common mismatch/insufficient cases.
    haystack = normalize(text)
    if part == "unknown" or issue == "unknown":
        if any(k in haystack for k in ["not inside", "missing contents", "not visible", "proof", "note says"]):
            status = "not_enough_information" if "not inside" in haystack else "contradicted"
            evidence = "false" if status == "not_enough_information" else evidence
            extra_flags.append("damage_not_visible")
    if "not working" in haystack or "stopped working" in haystack:
        issue = "none"
        status = "contradicted"
        extra_flags.append("damage_not_visible")
    if text_instruction_flag(text) and "note" in haystack and issue in {"water_damage", "crushed_packaging"}:
        extra_flags.append("claim_mismatch")

    flags = merge_flags(img_flags, exemplar_flags if score >= 0.42 else [], hist_flags, extra_flags)
    if status == "contradicted" and "claim_mismatch" not in flags and issue != "none":
        flags = merge_flags(flags.split(";"), ["claim_mismatch"])
    if status == "not_enough_information":
        flags = merge_flags(flags.split(";"), ["damage_not_visible"])
        evidence = "false"

    severity = severity_from_issue(issue, text, status)
    if exemplar and score >= 0.45 and exemplar.get("severity"):
        severity = exemplar["severity"]

    ids = image_ids(row.get("image_paths", ""))
    supporting = "none" if evidence == "false" or not ids else ids[0]
    if exemplar and score >= 0.40 and exemplar.get("supporting_image_ids") not in {"", "none"} and len(ids) > 1:
        # Preserve the useful pattern "second image is the close-up" without copying
        # a filename that belongs to a different folder.
        ex_idx = exemplar["supporting_image_ids"].split(";")[0]
        if ex_idx.startswith("img_"):
            wanted = ex_idx
            if wanted in ids:
                supporting = wanted

    if evidence == "true":
        evidence_reason = f"The submitted image set shows the claimed {part.replace('_', ' ')} clearly enough to evaluate the {issue.replace('_', ' ')} claim."
    else:
        evidence_reason = f"The submitted image set does not clearly show the claimed {part.replace('_', ' ')} condition."
    if status == "supported":
        justification = f"The available image evidence is consistent with the claimed {issue.replace('_', ' ')} on the {part.replace('_', ' ')}."
    elif status == "contradicted":
        justification = f"The images do not support the stated claim; the visible evidence differs from the claimed {issue.replace('_', ' ')} on the {part.replace('_', ' ')}."
    else:
        justification = f"The images are not sufficient to verify the claimed {issue.replace('_', ' ')} on the {part.replace('_', ' ')}."

    out = {col: row.get(col, "") for col in OUTPUT_COLUMNS}
    out.update(
        {
            "evidence_standard_met": evidence,
            "evidence_standard_met_reason": evidence_reason,
            "risk_flags": flags,
            "issue_type": issue,
            "object_part": part,
            "claim_status": status,
            "claim_status_justification": justification,
            "supporting_image_ids": supporting,
            "valid_image": valid_image,
            "severity": severity,
        }
    )
    return out


def run(input_csv: Path, output_csv: Path, repo_root: Path) -> List[Dict[str, str]]:
    dataset_root = repo_root / "dataset"
    rows = read_csv(input_csv)
    examples = read_csv(dataset_root / "sample_claims.csv")
    history = {r["user_id"]: r for r in read_csv(dataset_root / "user_history.csv")}
    predictions = [predict_row(row, dataset_root, history, examples) for row in rows]
    write_csv(output_csv, predictions)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HackerRank Orchestrate predictions.")
    parser.add_argument("--input", default="dataset/claims.csv", help="Input claims CSV path.")
    parser.add_argument("--output", default="output.csv", help="Output CSV path.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_csv = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    preds = run(input_csv, output_csv, repo_root)
    print(f"Wrote {len(preds)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
