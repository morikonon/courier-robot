#!/bin/bash
# Re-runs the experiments, renders report/report.pdf with Chrome, and builds the submission zip.
# Usage:  ./build_submission.sh <StudentID> <FullName>     e.g.  ./build_submission.sh 22B030000 Mukhamedali_Daniyaruly
set -e
cd "$(dirname "$0")"
ID="${1:-StudentID}"; NAME="${2:-FullName}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

python3 run_experiments.py
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/report/report.pdf" "file://$PWD/report/report.html" 2>/dev/null

ZIP="${ID}_${NAME}_CSCI3235_Midterm1.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" env.py agents.py run_experiments.py requirements.txt README.md \
  figures results report/report.pdf -x "*/__pycache__/*" "*.DS_Store"
echo "Built $ZIP"
grep -q "\[STUDENT ID\]\|\[WRITE WHAT\|\[week\]" report/report.html && \
  echo "WARNING: report/report.html still contains [placeholders]. Fill them and run this script again."
exit 0
