#!/usr/bin/env bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$BASE_DIR/reports/run_log.txt"
mkdir -p "$BASE_DIR/reports"
echo "RUN START $(date)" >> "$LOG"
echo "SEED=42" >> "$LOG"
steps=(
  "code/data_ingest.py"
  "code/preprocessing.py"
  "code/lexical_complexity.py"
  "code/readability_entropy.py"
  "code/sentiment_emotion.py"
  "code/discourse_markers_and_hedges.py"
  "code/marker_analysis.py"
  "code/topic_modeling.py"
  "code/embeddings_and_clustering.py"
  "code/statistical_tests_and_feature_importance.py"
  "code/reporting.py"
)

for s in "${steps[@]}"; do
  echo "-----" >> "$LOG"
  echo "START $s at $(date)" | tee -a "$LOG"
  python3 "$BASE_DIR/$s" 2>&1 | tee -a "$LOG" || echo "$s failed with exit code $?" | tee -a "$LOG"
  echo "END $s at $(date)" | tee -a "$LOG"
done

echo "RUN END $(date)" >> "$LOG"
echo "Packaging results..." | tee -a "$LOG"
cd "$BASE_DIR"
zip -r ../EDA-Rehydrated.zip . >/dev/null 2>&1 || echo "zip failed" | tee -a "$LOG"
echo "Archive created at ../EDA-Rehydrated.zip" | tee -a "$LOG"
echo "Summary appended." >> "$LOG"

echo "Done. See $LOG"
