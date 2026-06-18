import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd


OUTPUT_DIR = "/scratch/data/bikash_rs/Vivek/PRC-Emo"
IEMOCAP_TARGET_EMOTIONS = {
    "neu", "fru", "ang", "exc", "sad", "hap",
    "neutral", "frustrated", "angry", "excited", "sad", "happy",
}


def setup_logger(log_file=None):
    logger = logging.getLogger("iemocap_audio_desc_converter")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IEMOCAP openSMILE features to textual descriptions (JSON format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output_format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format: json or csv (default: json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split name used only for logging",
    )
    return parser.parse_args()


def extract_dialogue_info_from_row(row):
    dialogue_id = row.get("dialogue_id")
    utterance_id = row.get("utterance_id")

    if pd.isna(dialogue_id) or pd.isna(utterance_id):
        return None, None

    return str(dialogue_id), int(utterance_id)


def compute_thresholds(df, col, logger):
    try:
        low = df[col].quantile(0.33)
        high = df[col].quantile(0.66)
        logger.debug(f"{col}: low={low:.4f}, high={high:.4f}")
        return {"low": low, "high": high}
    except Exception as exc:
        logger.warning(f"Error computing thresholds for {col}: {exc}")
        return {"low": 0, "high": 1}


def describe(value, thresh, low_label, mid_label, high_label):
    if pd.isna(value):
        return mid_label
    if value < thresh["low"]:
        return low_label
    if value > thresh["high"]:
        return high_label
    return mid_label


def generate_description(row, thresholds):
    pitch = describe(
        row["F0semitoneFrom27.5Hz_sma3nz_amean"],
        thresholds["pitch"],
        "low-pitched",
        "moderate-pitched",
        "high-pitched",
    )

    variability = describe(
        row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
        thresholds["variability"],
        "stable",
        "moderately expressive",
        "highly expressive",
    )

    energy = describe(
        row["loudness_sma3_amean"],
        thresholds["loudness"],
        "low energy",
        "moderate energy",
        "high energy",
    )

    jitter_level = describe(
        row["jitterLocal_sma3nz_amean"],
        thresholds["jitter"],
        "stable voice",
        "slightly unstable voice",
        "shaky voice",
    )

    shimmer_level = describe(
        row["shimmerLocaldB_sma3nz_amean"],
        thresholds["shimmer"],
        "clear tone",
        "slightly rough tone",
        "rough tone",
    )

    noise = describe(
        row["spectralFlux_sma3_amean"],
        thresholds["spectral_flux"],
        "minimal background noise",
        "moderate background noise",
        "noticeable background noise",
    )

    return (
        f"{pitch}, {variability} speech with {energy}, "
        f"{jitter_level}, {shimmer_level}, and {noise}."
    )


def main():
    args = parse_args()
    input_file = args.input

    log_filename = f"iemocap_audio_desc_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = os.path.join(OUTPUT_DIR, log_filename)
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("IEMOCAP Audio Description Conversion Started")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Output format: {args.output_format}")

    try:
        df = pd.read_csv(input_file)
        logger.info(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.debug(f"Columns: {list(df.columns)}")
    except Exception as exc:
        logger.error(f"Failed to load CSV: {exc}")
        raise

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    required_cols = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "loudness_sma3_amean",
        "jitterLocal_sma3nz_amean",
        "shimmerLocaldB_sma3nz_amean",
        "spectralFlux_sma3_amean",
        "split",
        "session",
        "dialogue_id",
        "utterance_id",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    if "emotion" in df.columns:
        kept = df["emotion"].astype(str).str.lower().isin(IEMOCAP_TARGET_EMOTIONS)
        before = len(df)
        df = df.loc[kept].copy()
        logger.info(f"Filtered target emotions: kept {len(df)}/{before} rows")

    logger.info("Computing adaptive thresholds...")
    thresholds = {
        "pitch": compute_thresholds(df, "F0semitoneFrom27.5Hz_sma3nz_amean", logger),
        "variability": compute_thresholds(df, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", logger),
        "loudness": compute_thresholds(df, "loudness_sma3_amean", logger),
        "jitter": compute_thresholds(df, "jitterLocal_sma3nz_amean", logger),
        "shimmer": compute_thresholds(df, "shimmerLocaldB_sma3nz_amean", logger),
        "spectral_flux": compute_thresholds(df, "spectralFlux_sma3_amean", logger),
    }
    logger.info("✓ Thresholds computed")

    logger.info("Generating descriptions and grouping by dialogue_id...")
    dialogue_descriptions = {}
    error_count = 0
    success_count = 0

    for idx, row in df.iterrows():
        try:
            dialogue_id, utterance_idx = extract_dialogue_info_from_row(row)
            if dialogue_id is None:
                error_count += 1
                logger.warning(f"Skipping row {idx}: missing dialogue_id/utterance_id")
                continue

            description = generate_description(row, thresholds)

            if dialogue_id not in dialogue_descriptions:
                dialogue_descriptions[dialogue_id] = []

            while len(dialogue_descriptions[dialogue_id]) <= utterance_idx:
                dialogue_descriptions[dialogue_id].append(None)

            dialogue_descriptions[dialogue_id][utterance_idx] = description
            success_count += 1

        except Exception as exc:
            error_count += 1
            logger.error(f"Error processing row {idx}: {exc}")

    logger.info(f"✓ Processed {success_count} rows, {error_count} errors")
    logger.info(f"✓ Grouped into {len(dialogue_descriptions)} dialogues")

    final_output = {}
    for dialogue_id, descriptions in dialogue_descriptions.items():
        clean_descriptions = [d for d in descriptions if d is not None]
        if clean_descriptions:
            final_output[dialogue_id] = {
                "audio_descriptions": clean_descriptions,
            }

    logger.info(f"Final output has {len(final_output)} dialogues with descriptions")

    if args.output_format == "json":
        output_file = os.path.join(
            OUTPUT_DIR,
            os.path.basename(input_file).replace(".csv", "_audio_descriptions.json"),
        )

        logger.info(f"Saving to JSON: {output_file}")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Successfully saved to {output_file}")
            logger.info(f"✓ File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        except Exception as exc:
            logger.error(f"Failed to save JSON: {exc}")
            raise
    else:
        output_file = os.path.join(
            OUTPUT_DIR,
            os.path.basename(input_file).replace(".csv", "_audio_descriptions.csv"),
        )

        logger.info(f"Saving to CSV: {output_file}")
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Successfully saved to {output_file}")

    logger.info("=" * 80)
    logger.info("✓ Audio description conversion completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()