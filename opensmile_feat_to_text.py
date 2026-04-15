import pandas as pd
import argparse
import os
import json
import re
import logging
from datetime import datetime

# 🔹 Setup logging
def setup_logger(log_file=None):
    """Setup logger with both console and file handlers"""
    logger = logging.getLogger("audio_desc_converter")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger

# 🔹 Argument parser - MODIFIED TO ADD SPLIT PARAMETER
parser = argparse.ArgumentParser(description="Convert openSMILE features to text descriptions (JSON format)")
parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output_format", type=str, default="json", choices=["json", "csv"], 
                    help="Output format: json or csv (default: json)")
parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"],
                    help="Dataset split: train (no offset), dev (offset +1038), test (offset +1152)")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_DIR = "/scratch/data/bikash_rs/Vivek/PRC-Emo"

# 🔹 DEFINE OFFSETS FOR EACH SPLIT - NEW
DIALOGUE_OFFSETS = {
    "train": 0,
    "dev": 1039,
    "test": 1153
}
dialogue_offset = DIALOGUE_OFFSETS.get(args.split, 0)

# Setup logging
log_filename = f"audio_desc_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = os.path.join(OUTPUT_DIR, log_filename)
logger = setup_logger(log_file)

logger.info("="*80)
logger.info("Audio Description Conversion Started")
logger.info("="*80)
logger.info(f"Input file: {INPUT_FILE}")
logger.info(f"Dataset split: {args.split}")
logger.info(f"Dialogue ID offset: {dialogue_offset}")
logger.info(f"Output format: {args.output_format}")

# 🔹 Load data
try:
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")
except Exception as e:
    logger.error(f"Failed to load CSV: {e}")
    raise

# 🔹 Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Extract dialogue_id and utterance_idx from filename
def extract_dialogue_info(filename):
    """
    Extract dialogue_id and utterance_idx from filename
    Expected formats: 
    - dia577_utt6.wav → dialogue_id: '577', utterance_idx: 6
    - 1039_0.wav → dialogue_id: '1039', utterance_idx: 0
    """
    try:
        # Remove .wav extension if present
        name = filename.replace(".wav", "").replace(".csv", "")
        
        # Try format: dia<NUMBER>_utt<NUMBER>.wav
        if name.startswith("dia"):
            parts = name.split("_")
            if len(parts) >= 2:
                # Extract number from "dia577" → "577"
                dialogue_id = parts[0].replace("dia", "")
                
                # Extract number from "utt6" → "6"
                utt_part = parts[1]
                if utt_part.startswith("utt"):
                    utterance_idx = int(utt_part.replace("utt", ""))
                    return dialogue_id, utterance_idx
                else:
                    logger.warning(f"Unexpected utterance format in {filename}: {utt_part}")
                    return None, None
        
        # Fallback: try direct numeric format like 1039_0.wav
        else:
            parts = name.split("_")
            if len(parts) >= 2:
                try:
                    dialogue_id = parts[0]
                    utterance_idx = int(parts[1])
                    return dialogue_id, utterance_idx
                except ValueError:
                    logger.warning(f"Could not parse numeric format from {filename}")
                    return None, None
        
        logger.warning(f"Filename format unexpected: {filename}")
        return None, None
        
    except Exception as e:
        logger.warning(f"Error extracting dialogue info from {filename}: {e}")
        return None, None

# 🔹 ===== Adaptive Thresholds (based on percentiles) =====
logger.info("Computing adaptive thresholds...")

required_cols = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "spectralFlux_sma3_amean"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    logger.error(f"Missing required columns: {missing_cols}")
    logger.info(f"Available columns: {list(df.columns)}")
    raise ValueError(f"Missing required columns: {missing_cols}")

def compute_thresholds(df, col):
    """Compute low and high thresholds based on 33rd and 66th percentiles"""
    try:
        low = df[col].quantile(0.33)
        high = df[col].quantile(0.66)
        logger.debug(f"{col}: low={low:.4f}, high={high:.4f}")
        return {"low": low, "high": high}
    except Exception as e:
        logger.warning(f"Error computing thresholds for {col}: {e}")
        return {"low": 0, "high": 1}

thresholds = {
    "pitch": compute_thresholds(df, "F0semitoneFrom27.5Hz_sma3nz_amean"),
    "variability": compute_thresholds(df, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
    "loudness": compute_thresholds(df, "loudness_sma3_amean"),
    "jitter": compute_thresholds(df, "jitterLocal_sma3nz_amean"),
    "shimmer": compute_thresholds(df, "shimmerLocaldB_sma3nz_amean"),
    "spectral_flux": compute_thresholds(df, "spectralFlux_sma3_amean"),
}

logger.info("✓ Thresholds computed")

# 🔹 ===== Descriptor functions =====

def describe(value, thresh, low_label, mid_label, high_label):
    """Categorize value into low/medium/high based on thresholds"""
    if pd.isna(value):
        return mid_label
    if value < thresh["low"]:
        return low_label
    elif value > thresh["high"]:
        return high_label
    else:
        return mid_label


def generate_description(row):
    """Generate text description from audio features"""
    try:
        pitch = describe(
            row["F0semitoneFrom27.5Hz_sma3nz_amean"],
            thresholds["pitch"],
            "low-pitched",
            "moderate-pitched",
            "high-pitched"
        )

        variability = describe(
            row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
            thresholds["variability"],
            "stable",
            "moderately expressive",
            "highly expressive"
        )

        energy = describe(
            row["loudness_sma3_amean"],
            thresholds["loudness"],
            "low energy",
            "moderate energy",
            "high energy"
        )

        jitter_level = describe(
            row["jitterLocal_sma3nz_amean"],
            thresholds["jitter"],
            "stable voice",
            "slightly unstable voice",
            "shaky voice"
        )

        shimmer_level = describe(
            row["shimmerLocaldB_sma3nz_amean"],
            thresholds["shimmer"],
            "clear tone",
            "slightly rough tone",
            "rough tone"
        )

        noise = describe(
            row["spectralFlux_sma3_amean"],
            thresholds["spectral_flux"],
            "minimal background noise",
            "moderate background noise",
            "noticeable background noise"
        )

        # Combine intelligently (avoid repetition)
        description = (
            f"{pitch}, {variability} speech with {energy}, "
            f"{jitter_level}, {shimmer_level}, and {noise}."
        )

        return description

    except Exception as e:
        logger.warning(f"Error generating description: {e}")
        return "description unavailable"


# 🔹 Generate descriptions and group by dialogue
logger.info("Generating descriptions and grouping by dialogue...")

dialogue_descriptions = {}
error_count = 0
success_count = 0

for idx, row in df.iterrows():
    try:
        filename = row.get("file", f"unknown_{idx}")
        dialogue_id, utterance_idx = extract_dialogue_info(filename)
        
        if dialogue_id is None:
            error_count += 1
            logger.warning(f"Skipping row {idx}: Could not extract dialogue info from {filename}")
            continue
        
        # Generate description
        description = generate_description(row)
        
        # Group by dialogue_id
        if dialogue_id not in dialogue_descriptions:
            dialogue_descriptions[dialogue_id] = []
        
        # Ensure we have enough slots for this utterance
        while len(dialogue_descriptions[dialogue_id]) <= utterance_idx:
            dialogue_descriptions[dialogue_id].append(None)
        
        # Store description at correct index
        dialogue_descriptions[dialogue_id][utterance_idx] = description
        success_count += 1
        
    except Exception as e:
        error_count += 1
        logger.error(f"Error processing row {idx}: {e}")

logger.info(f"✓ Processed {success_count} rows, {error_count} errors")
logger.info(f"✓ Grouped into {len(dialogue_descriptions)} dialogues")

# 🔹 Clean up None values and prepare final output - MODIFIED TO APPLY OFFSET
final_output = {}
for dialogue_id, descriptions in dialogue_descriptions.items():
    # Remove None values but keep order
    clean_descriptions = [d for d in descriptions if d is not None]
    if clean_descriptions:
        # APPLY DIALOGUE OFFSET - NEW
        adjusted_dialogue_id = str(int(dialogue_id) + dialogue_offset)
        final_output[adjusted_dialogue_id] = {
            "audio_descriptions": clean_descriptions
        }

logger.info(f"Final output has {len(final_output)} dialogues with descriptions")
if dialogue_offset > 0:
    logger.info(f"Applied dialogue ID offset: +{dialogue_offset}")

# 🔹 Save output
if args.output_format == "json":
    output_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(INPUT_FILE).replace(".csv", "_audio_descriptions.json")
    )
    
    logger.info(f"Saving to JSON: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Successfully saved to {output_file}")
        logger.info(f"✓ File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise

else:
    # CSV format - flatten the structure
    output_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(INPUT_FILE).replace(".csv", "_audio_descriptions.csv")
    )
    
    logger.info(f"Saving to CSV: {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Successfully saved to {output_file}")

logger.info("="*80)
logger.info("✓ Audio description conversion completed successfully!")
logger.info("="*80)