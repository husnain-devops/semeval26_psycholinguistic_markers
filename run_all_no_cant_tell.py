#!/usr/bin/env python3
"""
Run the EDA pipeline on a version of the data where all 'cant_tell' labels
have been removed (binary yes/no only).

This script first runs the custom data_ingest in EDA-Rehydrated/code_no_cant_tell
to filter out 'cant_tell', then runs the standard EDA-Rehydrated/code scripts
to generate figures and reports based on the filtered dataset.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
EDA_DIR = BASE_DIR / "EDA-Rehydrated"
CODE_DIR_STD = EDA_DIR / "code"
CODE_DIR_BIN = EDA_DIR / "code_no_cant_tell"
LOG_DIR = EDA_DIR / "reports"
LOG_FILE = LOG_DIR / "run_log_no_cant_tell.txt"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def append_log(msg: str):
    """Append message to log file and print it."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def get_script_plan() -> List[Tuple[Path, str]]:
    """
    Return an ordered list of (directory, script_name) pairs to execute.
    
    The first step uses the binary-only data_ingest in CODE_DIR_BIN to
    drop 'cant_tell'. All subsequent steps use the standard code folder,
    which will then operate on the filtered dataset.
    """
    plan: List[Tuple[Path, str]] = [
        (CODE_DIR_BIN, "data_ingest.py"),
        (CODE_DIR_STD, "preprocessing.py"),
        (CODE_DIR_STD, "pos_analysis.py"),
        (CODE_DIR_STD, "lexical_complexity.py"),
        (CODE_DIR_STD, "ma_ttr_mtld_analysis.py"),
        (CODE_DIR_STD, "readability_entropy.py"),
        (CODE_DIR_STD, "plot_readability.py"),
        (CODE_DIR_STD, "sentiment_emotion.py"),
        (CODE_DIR_STD, "discourse_markers_and_hedges.py"),
        (CODE_DIR_STD, "marker_analysis.py"),
        (CODE_DIR_STD, "topic_modeling.py"),
        (CODE_DIR_STD, "embeddings_and_clustering.py"),
        (CODE_DIR_STD, "statistical_tests_and_feature_importance.py"),
        (CODE_DIR_STD, "reporting.py"),
    ]
    return plan

def run_script(script_dir: Path, script_name: str) -> bool:
    """
    Run a single Python script from the given directory.
    Returns True if successful.
    """
    script_path = script_dir / script_name
    append_log("=" * 60)
    append_log(f"START {script_name} ({script_dir.name}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    append_log("=" * 60)
    
    if not script_path.exists():
        append_log(f"✗ {script_name} not found in {script_dir}")
        append_log(f"END {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        append_log("")
        return False
    
    try:
        original_cwd = os.getcwd()
        # Change to EDA-Rehydrated directory so scripts see the expected layout
        os.chdir(EDA_DIR)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=EDA_DIR,
            capture_output=False,
            text=True,
            check=False,
        )
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            append_log(f"✓ {script_name} completed successfully")
            success = True
        else:
            append_log(f"✗ {script_name} failed with exit code {result.returncode}")
            success = False
    except Exception as e:
        append_log(f"✗ {script_name} failed with exception: {str(e)}")
        success = False
    finally:
        append_log(f"END {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        append_log("")
    
    return success

def main():
    """Main function to run the binary-only EDA pipeline."""
    append_log("=" * 60)
    append_log("RUN (no_cant_tell) START " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    append_log("SEED=42")
    append_log(f"EDA directory: {EDA_DIR}")
    append_log(f"Binary ingest dir: {CODE_DIR_BIN}")
    append_log(f"Standard code dir: {CODE_DIR_STD}")
    append_log("=" * 60)
    
    # Check directories
    if not CODE_DIR_BIN.exists():
        append_log(f"ERROR: Binary code directory not found: {CODE_DIR_BIN}")
        sys.exit(1)
    if not CODE_DIR_STD.exists():
        append_log(f"ERROR: Standard code directory not found: {CODE_DIR_STD}")
        sys.exit(1)
    
    plan = get_script_plan()
    append_log(f"Found {len(plan)} step(s) to run:")
    for script_dir, script_name in plan:
        append_log(f"  - {script_dir.name}/{script_name}")
    append_log("")
    
    results = {}
    for script_dir, script_name in plan:
        success = run_script(script_dir, script_name)
        results[f"{script_dir.name}/{script_name}"] = success
    
    # Summary
    append_log("=" * 60)
    append_log("RUN (no_cant_tell) SUMMARY")
    append_log("=" * 60)
    successful = sum(1 for s in results.values() if s)
    failed = len(results) - successful
    
    for key, success in results.items():
        status = "✓" if success else "✗"
        append_log(f"{status} {key}")
    
    append_log("")
    append_log(f"Total: {len(results)} steps")
    append_log(f"Successful: {successful}")
    append_log(f"Failed: {failed}")
    append_log("=" * 60)
    append_log("RUN (no_cant_tell) END " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    append_log("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

