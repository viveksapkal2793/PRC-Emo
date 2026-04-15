import os
import argparse
import logging
import pandas as pd
import opensmile
from tqdm import tqdm
from datetime import datetime

# 🔹 Setup logging
def setup_logger(log_file=None):
    """Setup logger with both console and file handlers"""
    logger = logging.getLogger("opensmile_extractor")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# 🔹 Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract audio features using openSMILE and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python opensmile_ext.py --audio_dir /path/to/audio/dir --output_csv features.csv
  python opensmile_ext.py -d /scratch/data/bikash_rs/Vivek/dataset/MELD_audio/train/ -o train_features.csv
        """
    )
    
    parser.add_argument(
        '--audio_dir', '-d',
        type=str,
        required=True,
        help='Path to directory containing .wav audio files'
    )
    
    parser.add_argument(
        '--output_csv', '-o',
        type=str,
        required=True,
        help='Output CSV filename (will be saved in /scratch/data/bikash_rs/Vivek/PRC-Emo)'
    )
    
    parser.add_argument(
        '--save_log',
        type=bool,
        default=True,
        help='Save processing log to file (default: True)'
    )
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup paths
    BASE_OUTPUT_DIR = "/scratch/data/bikash_rs/Vivek/PRC-Emo"
    audio_dir = args.audio_dir
    output_csv_name = args.output_csv
    output_csv_path = os.path.join(BASE_OUTPUT_DIR, output_csv_name)
    
    # Setup logging
    log_file = None
    if args.save_log:
        log_filename = f"opensmile_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = os.path.join(BASE_OUTPUT_DIR, log_filename)
    
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("OpenSMILE Feature Extraction Started")
    logger.info("="*80)
    
    # Validate inputs
    if not os.path.exists(audio_dir):
        logger.error(f"Audio directory not found: {audio_dir}")
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    if not os.path.exists(BASE_OUTPUT_DIR):
        logger.error(f"Output directory not found: {BASE_OUTPUT_DIR}")
        raise FileNotFoundError(f"Output directory not found: {BASE_OUTPUT_DIR}")
    
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Output CSV path: {output_csv_path}")
    logger.info(f"Log file: {log_file if log_file else 'Console only'}")
    
    # 🔹 Initialize openSMILE
    logger.info("Initializing openSMILE with eGeMAPSv02 feature set...")
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        logger.info("✓ openSMILE initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize openSMILE: {e}")
        raise
    
    # 🔹 Collect all audio files
    logger.info(f"Scanning audio directory: {audio_dir}")
    audio_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ]
    
    logger.info(f"Found {len(audio_files)} .wav audio files")
    
    if len(audio_files) == 0:
        logger.warning("No .wav files found in the specified directory!")
        return
    
    # 🔹 Process files
    logger.info("="*80)
    logger.info("Starting audio feature extraction...")
    logger.info("="*80)
    
    all_features = []
    error_count = 0
    success_count = 0
    
    for idx, file in enumerate(tqdm(audio_files, desc="Processing audio files"), 1):
        try:
            logger.debug(f"Processing [{idx}/{len(audio_files)}]: {os.path.basename(file)}")
            df = smile.process_file(file)
            
            # Add filename column (important)
            df["file"] = os.path.basename(file)
            
            all_features.append(df)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {os.path.basename(file)}: {str(e)}")
    
    logger.info("="*80)
    logger.info(f"Processing Summary: {success_count} successful, {error_count} failed out of {len(audio_files)} files")
    logger.info("="*80)
    
    if len(all_features) == 0:
        logger.error("No features were successfully extracted!")
        raise RuntimeError("No features were successfully extracted!")
    
    # 🔹 Combine all results
    logger.info("Combining all extracted features...")
    final_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"✓ Combined features shape: {final_df.shape} (rows, columns)")
    
    # 🔹 Save to CSV
    logger.info(f"Saving features to CSV: {output_csv_path}")
    try:
        final_df.to_csv(output_csv_path, index=False)
        logger.info(f"✓ Successfully saved {len(final_df)} feature vectors to {output_csv_path}")
        logger.info(f"✓ CSV file size: {os.path.getsize(output_csv_path) / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")
        raise
    
    logger.info("="*80)
    logger.info("✓ Feature extraction completed successfully!")
    logger.info("="*80)

if __name__ == "__main__":
    main()