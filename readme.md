# IndiaRuns - Redrob Candidate Ranking

This repository contains a standalone ranking pipeline for the Redrob hackathon challenge. The goal is to score and rank candidate profiles against the provided job description using a mix of keyword, semantic, behavioral, and quality signals.

## Project Overview

- Uses the candidate dataset from the challenge bundle.
- Builds a weighted composite score for each candidate.
- Generates a concise, fact-based reasoning string for each top-ranked candidate.
- Produces a submission CSV in the exact format required by the challenge.

## Files in this Repository

- `rank.py` — main ranking script
- `requirements.txt` — Python dependencies
- `submission.csv` — generated submission output
- `submission_metadata.yaml` — submission metadata template
- `redrob_ranking_system.ipynb` — exploratory notebook used to develop the pipeline

## Setup

1. Create and activate a Python environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Reproduce the Submission

Run the ranking script with the challenge dataset:

```bash
python rank.py \
  --candidates "./candidates.jsonl" \
  --team-id NoBlackBox \
  --out ./submission.csv
```

This will generate a CSV with the top 100 ranked candidates and the required columns:

- `candidate_id`
- `rank`
- `score`
- `reasoning`

## Optional Validation

You can validate the output file using the challenge checker:

```bash
python "./validate_submission.py" ./submission.csv
```

## Ranking Approach (High Level)

The pipeline performs the following steps:

1. Loads and validates candidate records.
2. Detects suspicious or low-quality profiles (honeypot checks).
3. Builds text features from resumes, titles, skills, and career history.
4. Computes semantic similarity using TF-IDF and BM25-style matching against the job description.
5. Applies weighted scoring across skills, experience, location, role fit, and behavioral signals.
6. Sorts candidates, applies tie-break logic, and writes the final submission CSV.

## Notes

- No external APIs are used during ranking.
- The script is designed to run locally on CPU within the challenge constraints.
- The output is intended to match the required submission schema exactly.
