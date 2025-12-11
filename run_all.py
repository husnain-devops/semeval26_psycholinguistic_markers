#!/usr/bin/env python3
"""
Run all Python scripts in the EDA-Rehydrated/code directory.
Excludes data_processed.py as it has been moved to a different location.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "EDA-Rehydrated" / "code"
LOG_DIR = BASE_DIR / "EDA-Rehydrated" / "reports"
LOG_FILE = LOG_DIR / "run_log.txt"

# Scripts to exclude
EXCLUDED_SCRIPTS = {"data_processed.py"}

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

def get_script_order() -> List[Path]:
    """
    Get scripts in a logical execution order.
    This order ensures dependencies are met (e.g., data_ingest before preprocessing).
    Excludes data_processed.py as it has been moved.
    """
    # Define the preferred execution order based on dependencies
    preferred_order = [
        "data_ingest.py",
        "preprocessing.py",
        "pos_analysis.py",
        "lexical_complexity.py",
        "ma_ttr_mtld_analysis.py",
        "readability_entropy.py",
        "plot_readability.py",
        "sentiment_emotion.py",
        "discourse_markers_and_hedges.py",
        "marker_analysis.py",
        "topic_modeling.py",
        "embeddings_and_clustering.py",
        "statistical_tests_and_feature_importance.py",
        "reporting.py",
    ]
    
    # Get all Python scripts in the code directory, excluding specified ones
    all_scripts = [
        f for f in CODE_DIR.glob("*.py") 
        if f.is_file() and f.name not in EXCLUDED_SCRIPTS
    ]
    
    # Create ordered list: first preferred order, then any remaining scripts
    ordered_scripts = []
    script_names = {s.name for s in all_scripts}
    
    # Add scripts in preferred order
    for script_name in preferred_order:
        if script_name in EXCLUDED_SCRIPTS:
            continue
        script_path = CODE_DIR / script_name
        if script_path.exists() and script_path.is_file():
            ordered_scripts.append(script_path)
            script_names.discard(script_name)
    
    # Add any remaining scripts not in preferred order
    for script in sorted(all_scripts):
        if script.name in script_names:
            ordered_scripts.append(script)
    
    return ordered_scripts

def run_script(script_path: Path) -> bool:
    """
    Run a single Python script and return True if successful.
    """
    script_name = script_path.name
    append_log("=" * 60)
    append_log(f"START {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    append_log("=" * 60)
    
    try:
        # Change to the code directory to ensure relative imports work
        original_cwd = os.getcwd()
        os.chdir(CODE_DIR.parent)  # Change to EDA-Rehydrated directory
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=CODE_DIR.parent,
            capture_output=False,  # Let output go to stdout/stderr
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            append_log(f"✓ {script_name} completed successfully")
            return True
        else:
            append_log(f"✗ {script_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        append_log(f"✗ {script_name} failed with exception: {str(e)}")
        os.chdir(original_cwd)
        return False
    finally:
        append_log(f"END {script_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        append_log("")

def main():
    """Main function to run all scripts."""
    append_log("=" * 60)
    append_log("RUN START " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    append_log("SEED=42")
    append_log(f"Code directory: {CODE_DIR}")
    if EXCLUDED_SCRIPTS:
        append_log(f"Excluded scripts: {', '.join(EXCLUDED_SCRIPTS)}")
    append_log("=" * 60)
    
    # Check if code directory exists
    if not CODE_DIR.exists():
        append_log(f"ERROR: Code directory not found: {CODE_DIR}")
        sys.exit(1)
    
    # Get all scripts to run
    scripts = get_script_order()
    
    if not scripts:
        append_log("WARNING: No Python scripts found in code directory")
        sys.exit(0)
    
    append_log(f"Found {len(scripts)} script(s) to run:")
    for script in scripts:
        append_log(f"  - {script.name}")
    append_log("")
    
    # Run each script
    results = {}
    for script in scripts:
        success = run_script(script)
        results[script.name] = success
    
    # Print summary
    append_log("=" * 60)
    append_log("RUN SUMMARY")
    append_log("=" * 60)
    successful = sum(1 for s in results.values() if s)
    failed = len(results) - successful
    
    for script_name, success in results.items():
        status = "✓" if success else "✗"
        append_log(f"{status} {script_name}")
    
    append_log("")
    append_log(f"Total: {len(results)} scripts")
    append_log(f"Successful: {successful}")
    append_log(f"Failed: {failed}")
    append_log("=" * 60)
    append_log("RUN END " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    append_log("=" * 60)
    
    # Exit with error code if any script failed
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

