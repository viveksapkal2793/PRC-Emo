import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd


OUTPUT_DIR = "/scratch/data/bikash_rs/Vivek/PRC-Emo"

IEMOCAP_TARGET_EMOTIONS = {
    "neu", "fru", "ang", "exc", "sad", "hap",
    "neutral", "frustrated", "angry", "excited", "happy"
}

OPENFACE_ROOT = "/scratch/data/bikash_rs/Vivek/dataset/IEMOCAP_openface_aus"

AU_NAMES = {
    "AU01": "Inner brow raiser",
    "AU02": "Outer brow raiser",
    "AU04": "Brow lowerer",
    "AU05": "Upper lid raiser",
    "AU06": "Cheek raiser",
    "AU07": "Lid tightener",
    "AU09": "Nose wrinkler",
    "AU10": "Upper lip raiser",
    "AU12": "Lip corner puller",
    "AU14": "Dimpler",
    "AU15": "Lip corner depressor",
    "AU17": "Chin raiser",
    "AU20": "Lip stretcher",
    "AU23": "Lip tightener",
    "AU25": "Lips part",
    "AU26": "Jaw drop",
    "AU28": "Lip suck",
    "AU45": "Blink",
}

TARGET_AU_COLS = [f"{k}_r" for k in AU_NAMES.keys()]


def setup_logger(log_file=None):
    logger = logging.getLogger("iemocap_visual_desc_converter")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    return parser.parse_args()


def normalize_intensity(intensity):
    if pd.isna(intensity):
        return 0.0

    intensity = float(intensity)

    if intensity > 5.0:
        intensity_str = f"{intensity:.2f}"

        if "." in intensity_str:
            parts = intensity_str.split(".")
            if len(parts[0]) > 1:
                normalized = float(f"{parts[0][-1]}.{parts[1]}")
            else:
                normalized = float(f"{parts[0]}.{parts[1]}")
        else:
            normalized = float(intensity_str[-1])

        return min(normalized, 5.0)

    return intensity


def is_all_zero(df):
    existing_cols = [c for c in TARGET_AU_COLS if c in df.columns]

    if not existing_cols:
        return True

    values = df[existing_cols].fillna(0).values

    return (np.abs(values) < 1e-6).all()


def get_peak_frame_description(csv_path, threshold=0.8):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if df.empty:
            return "No visual expressions detected."

        if is_all_zero(df):
            return "No visual expressions detected."

        au_cols = [c for c in df.columns if c.endswith("_r")]

        if not au_cols:
            return "No visual expressions detected."

        df["overall_intensity"] = df[au_cols].sum(axis=1)

        peak_idx = df["overall_intensity"].idxmax()
        peak_frame = df.loc[peak_idx]

        active_aus = []

        for au_code, au_name in AU_NAMES.items():
            col = f"{au_code}_r"

            if col not in peak_frame:
                continue

            intensity = normalize_intensity(peak_frame[col])

            if intensity >= threshold:
                active_aus.append(
                    (au_name, intensity)
                )

        active_aus.sort(key=lambda x: x[1], reverse=True)

        if not active_aus:
            return "No significant facial expressions detected."

        return ", ".join(
            f"{name} (intensity: {val:.2f})"
            for name, val in active_aus
        )

    except Exception:
        return "Error processing visual features."


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_filename = (
        f"iemocap_visual_desc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = setup_logger(os.path.join(OUTPUT_DIR, log_filename))

    logger.info("Loading metadata CSV...")
    df = pd.read_csv(args.input)

    df = df[
        df["emotion"].astype(str).str.lower().isin(IEMOCAP_TARGET_EMOTIONS)
    ].copy()

    logger.info(f"Filtered rows: {len(df)}")

    dialogue_descriptions = {}
    success_count = 0
    error_count = 0

    for _, row in df.iterrows():
        dialogue_id = str(row["dialogue_id"])
        utterance_id = int(row["utterance_id"])
        turn_name = str(row["turn_name"])

        csv_path = os.path.join(
            OPENFACE_ROOT,
            args.split,
            f"{turn_name}.csv"
        )

        description = get_peak_frame_description(csv_path)

        if dialogue_id not in dialogue_descriptions:
            dialogue_descriptions[dialogue_id] = []

        while len(dialogue_descriptions[dialogue_id]) <= utterance_id:
            dialogue_descriptions[dialogue_id].append(None)

        dialogue_descriptions[dialogue_id][utterance_id] = description
        success_count += 1

    final_output = {}

    for dialogue_id, descriptions in dialogue_descriptions.items():
        clean_descriptions = [d for d in descriptions if d is not None]

        if clean_descriptions:
            final_output[dialogue_id] = {
                "visual_descriptions": clean_descriptions
            }

    output_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(args.input).replace(
            ".csv",
            "_visual_descriptions.json"
        )
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to: {output_file}")
    logger.info(f"Processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Dialogues: {len(final_output)}")


if __name__ == "__main__":
    main()