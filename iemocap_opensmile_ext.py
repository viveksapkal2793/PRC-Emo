import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set

import opensmile
import pandas as pd
from tqdm import tqdm


TARGET_EMOTIONS = {"neu", "fru", "ang", "exc", "sad", "hap"}


@dataclass(frozen=True)
class UtteranceRecord:
    split: str
    session: str
    dialogue_id: str
    utterance_id: int
    turn_name: str
    emotion: str
    wav_path: Path


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("iemocap_opensmile_extractor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract utterance-wise openSMILE features for IEMOCAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iemocap_root",
        type=str,
        default="/scratch/data/bikash_rs/Vivek/dataset/IEMOCAP_full_release",
        help="Path to the IEMOCAP_full_release directory",
    )
    parser.add_argument(
        "--valid_json",
        type=str,
        default="/scratch/data/bikash_rs/Vivek/PRC-Emo/data/iemocap.valid.json",
        help="Path to the validation dialogue-id JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/data/bikash_rs/Vivek/PRC-Emo",
        help="Directory to save the output CSV files",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="eGeMAPSv02",
        help="openSMILE feature set name",
    )
    parser.add_argument(
        "--feature_level",
        type=str,
        default="Functionals",
        help="openSMILE feature level name",
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="Write a timestamped log file alongside the CSV outputs",
    )
    return parser.parse_args()


def load_valid_dialogue_ids(valid_json_path: str) -> Set[str]:
    with open(valid_json_path, "r", encoding="utf-8") as f:
        valid_data = json.load(f)
    if isinstance(valid_data, dict):
        return set(valid_data.keys())
    if isinstance(valid_data, list):
        result = set()
        for item in valid_data:
            if isinstance(item, dict) and "s_id" in item:
                result.add(str(item["s_id"]))
        return result
    raise ValueError(f"Unsupported validation JSON format in {valid_json_path}")


def locate_emoeval_file(iemocap_root: Path, session_num: int, dialogue_id: str) -> Path:
    session_dir = iemocap_root / f"Session{session_num}" / "dialog" / "EmoEvaluation"
    candidate = session_dir / f"{dialogue_id}.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"EmoEvaluation file not found for {dialogue_id}: {candidate}")


def resolve_utterance_wav(iemocap_root: Path, session_num: int, dialogue_id: str, turn_name: str) -> Path:
    wav_dir = iemocap_root / f"Session{session_num}" / "sentences" / "wav" / dialogue_id
    wav_path = wav_dir / f"{turn_name}.wav"
    if wav_path.exists():
        return wav_path
    raise FileNotFoundError(f"Wav file not found: {wav_path}")


def parse_emoeval_file(
    iemocap_root: Path,
    session_num: int,
    dialogue_id: str,
    split: str,
) -> List[UtteranceRecord]:
    emoeval_path = locate_emoeval_file(iemocap_root, session_num, dialogue_id)
    records: List[UtteranceRecord] = []
    utterance_index = 0

    with open(emoeval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue

            match = re.match(
                r"^\[(?P<start>[0-9.]+) - (?P<end>[0-9.]+)\]\s+(?P<turn>\S+)\s+(?P<emotion>\S+)\s+\[(?P<vad>.+)\]$",
                line,
            )
            if match is None:
                continue

            turn_name = match.group("turn")
            emotion = match.group("emotion").lower()
            if emotion not in TARGET_EMOTIONS:
                continue

            wav_path = resolve_utterance_wav(iemocap_root, session_num, dialogue_id, turn_name)
            records.append(
                UtteranceRecord(
                    split=split,
                    session=f"Session{session_num}",
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_index,
                    turn_name=turn_name,
                    emotion=emotion,
                    wav_path=wav_path,
                )
            )
            utterance_index += 1

    return records


def collect_dialogue_ids_for_session(session_dir: Path) -> List[str]:
    emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
    return [txt_file.stem for txt_file in sorted(emo_eval_dir.glob("*.txt"))]


def build_split_records(
    iemocap_root: Path,
    split: str,
    session_nums: Iterable[int],
    valid_dialogue_ids: Set[str],
) -> List[UtteranceRecord]:
    records: List[UtteranceRecord] = []

    for session_num in session_nums:
        session_dir = iemocap_root / f"Session{session_num}"
        dialogue_ids = collect_dialogue_ids_for_session(session_dir)

        for dialogue_id in dialogue_ids:
            if split == "valid" and dialogue_id not in valid_dialogue_ids:
                continue
            if split == "train" and dialogue_id in valid_dialogue_ids:
                continue

            records.extend(
                parse_emoeval_file(
                    iemocap_root=iemocap_root,
                    session_num=session_num,
                    dialogue_id=dialogue_id,
                    split=split,
                )
            )

    return records


def extract_features(records: List[UtteranceRecord], smile: opensmile.Smile, logger: logging.Logger) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    errors = 0

    for record in tqdm(records, desc="Extracting utterances"):
        try:
            df = smile.process_file(str(record.wav_path)).reset_index(drop=True)
            df["split"] = record.split
            df["session"] = record.session
            df["dialogue_id"] = record.dialogue_id
            df["utterance_id"] = record.utterance_id
            df["turn_name"] = record.turn_name
            df["emotion"] = record.emotion
            df["wav_path"] = str(record.wav_path)
            rows.append(df)
        except Exception as exc:
            errors += 1
            logger.error(f"Failed to process {record.wav_path}: {exc}")

    if not rows:
        raise RuntimeError("No utterance features were extracted successfully")

    if errors:
        logger.warning(f"Encountered {errors} extraction failures")

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    iemocap_root = Path(args.iemocap_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if args.save_log:
        log_file = str(output_dir / f"iemocap_opensmile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = setup_logger(log_file)
    valid_dialogue_ids = load_valid_dialogue_ids(args.valid_json)

    logger.info("=" * 80)
    logger.info("IEMOCAP utterance-wise openSMILE extraction started")
    logger.info(f"IEMOCAP root: {iemocap_root}")
    logger.info(f"Valid dialogue IDs: {len(valid_dialogue_ids)}")
    logger.info("Target emotions: %s", ", ".join(sorted(TARGET_EMOTIONS)))
    logger.info("=" * 80)

    feature_set = getattr(opensmile.FeatureSet, args.feature_set)
    feature_level = getattr(opensmile.FeatureLevel, args.feature_level)
    smile = opensmile.Smile(feature_set=feature_set, feature_level=feature_level)

    split_configs = {
        "train": [1, 2, 3, 4],
        "valid": [1, 2, 3, 4],
        "test": [5],
    }

    for split, session_nums in split_configs.items():
        logger.info(f"Building {split} records from sessions {session_nums}")
        records = build_split_records(
            iemocap_root=iemocap_root,
            split=split,
            session_nums=session_nums,
            valid_dialogue_ids=valid_dialogue_ids,
        )

        logger.info(f"{split}: collected {len(records)} utterances")
        if not records:
            logger.warning(f"{split}: no records found, skipping")
            continue

        features_df = extract_features(records, smile, logger)
        output_csv = output_dir / f"iemocap_{split}_opensmile_utterance_wise.csv"
        features_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(features_df)} rows to {output_csv}")

    logger.info("Extraction completed successfully")


if __name__ == "__main__":
    main()