#!/bin/bash

START_DATE="2025-12-01"
END_DATE=$(date +"%Y-%m-%d")

current="$START_DATE"

while [[ "$current" < "$END_DATE" || "$current" == "$END_DATE" ]]; do
    echo "Commit for $current"

    echo "Daily activity for $current" > history_$current.txt

    git add .
    
    GIT_AUTHOR_DATE="$current 12:00:00" \
    GIT_COMMITTER_DATE="$current 12:00:00" \
    git commit -m "Daily auto commit for $current"

    # Increment by 1 day
    current=$(date -I -d "$current + 1 day")
done

