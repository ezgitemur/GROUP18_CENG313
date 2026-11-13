# GROUP18_CENG313: CMI – Detect Behavior with Sensor Data

Course project for CENG313 Introduction to Data Science, based on the Kaggle competition
[CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).

**Group ID:** 18

## Team Members

| Name Surname | Student ID | GitHub Username |
|--------------|------------|-----------------|
| Ezgi Nur Temür | 23118080047 | [@ezgitemur](https://github.com/ezgitemur) |
| İrem Su Tepedelen | 23118080088 | [@1rems](https://github.com/1rems) |

## Competition Overview

The goal is to detect body-focused repetitive behaviors (BFRBs), such as hair pulling and skin picking,
using data from a wrist-worn device (Helios). Each recording (sequence) must be classified into one of
18 gestures: 8 BFRB gestures and 10 non-BFRB gestures.

**Evaluation metric:** the average of
- binary F1 (BFRB vs. non-BFRB), and
- macro F1 over the 8 BFRB gestures plus a single merged non-BFRB class.

About half of the hidden test set contains only IMU data, so models must also work without the
thermopile and ToF sensors.

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | Sensor readings, one row per time step (574,945 rows, 341 columns) |
| `train_demographics.csv` | Participant information (81 participants) |
| `test.csv`, `test_demographics.csv` | Small sample files showing the test format |

Sensor columns:
- **IMU (7):** acceleration (`acc_x/y/z`) and orientation quaternion (`rot_w/x/y/z`)
- **Thermopile (5):** temperature sensors (`thm_1`–`thm_5`)
- **ToF (320):** 5 time-of-flight sensors with 8×8 pixels each (`tof_1_v0`–`tof_5_v63`)

The data is not included in this repository (Kaggle rules and file size). It is downloaded into
`data/` via the Kaggle API inside the notebook.

## Approach

### Assignment 1: EDA and Baselines

**Exploratory data analysis** covers dataset structure, data types, missing values, ToF "no detection" (-1)
values, duplicates, gesture distribution, sequence lengths, example sensor signals, and participant demographics.

Key findings:
- 8,151 sequences from 81 subjects; BFRB gestures are balanced, non-BFRB gestures are imbalanced.
- Sequence lengths are right-skewed (median 59, max 700 time steps).
- 57.5% of ToF pixel values are -1 (no detection), less often in BFRB sequences (50.7%)
  than in non-BFRB sequences (67.6%), so -1 values carry information.

**Preprocessing:** each sequence is summarized by the mean, std, min, and max of each sensor channel.
ToF pixels are first summarized per time step as the mean distance of detected pixels and the fraction
of pixels with no detection. Remaining missing values are filled with the column median inside the
model pipeline, fitted on the training folds only, so no information from held-out participants leaks
into training.

**Validation:** 5-fold `GroupKFold` by subject, so that each validation fold contains only unseen participants.

### Assignment 2: Proposed Solution

*To be added.*

## Results

| Model | Features | CV score (mean) | CV score (std) |
|-------|----------|-----------------|----------------|
| Majority class | 0 | 0.3936 | – |
| Random Forest (IMU only) | 28 | 0.6516 | 0.0285 |
| Random Forest (all sensors) | 88 | 0.7200 | 0.0198 |

## Kaggle Score

| Date | Model | Public Score | Private Score |
|------|-------|--------------|---------------|
| – | – | – | – |

*Not yet submitted. This is a code competition, so submissions must run through the competition's
inference server in a Kaggle notebook. A late submission of the all-sensor random forest is planned;
its scores and a screenshot (in `submissions/`) will be added here.*

## How to Run

1. Create a Kaggle account, open the competition page, and accept the rules
   (via **Late Submission**).
2. In Kaggle, go to **Settings → API → Create Legacy API Key** to download `kaggle.json`.
3. Open `assignment1_eda_baseline.ipynb` in Google Colab
   (**File → Open notebook → GitHub**, then select this repository).
4. In Colab, open the **Secrets** panel (key icon) and add:
   - `KAGGLE_USERNAME`: the `username` value from `kaggle.json`
   - `KAGGLE_KEY`: the `key` value from `kaggle.json`
5. Run all cells (**Runtime → Run all**). The first cells download the data into `data/` and clone
   this repository to import the code in `src/`.

The required packages are listed in `requirements.txt`. They are preinstalled in Colab; to run
locally, use `pip install -r requirements.txt`.

## Repository Structure

```
GROUP18_CENG313/
├── README.md
├── requirements.txt                 # Python packages used
├── .gitignore                       # excludes data/ and credentials
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA and baseline models
├── assignment2_solution.ipynb       # Assignment 2: proposed solution (to be added)
├── src/
│   ├── __init__.py
│   ├── data.py                      # Data loading, sensor grouping, feature extraction
│   ├── models.py                    # Model definitions and GroupKFold cross-validation
│   └── utils.py                     # Competition metric
├── data/                            # Not committed; downloaded via the Kaggle API
└── submissions/                     # Kaggle score screenshots
```

## Progress Log

| Date | Member | Progress |
|------|--------|----------|
| 2026-11-03 | Ezgi Nur Temür | Repository setup, `.gitignore` for data, Kaggle credentials and model files |
| 2026-11-08 | Ezgi Nur Temür | Initial EDA: data loading, class distribution, sequence lengths |
| 2026-11-09 | Ezgi Nur Temür | EDA plots, missing values, ToF analysis, example sensor signals, demographics |
| 2026-11-09 | İrem Su Tepedelen | Data loading and preprocessing utilities in `src/data.py` |
| 2026-11-10 | Ezgi Nur Temür | Sequence labels and feature extraction, competition metric (`src/utils.py`), GroupKFold cross-validation and random forest (`src/models.py`), ToF summarization, baseline models and comparison table |
| 2026-11-11 | Ezgi Nur Temür | README with project details and instructions |
| 2026-11-13 | İrem Su Tepedelen | `src/__init__.py`, `requirements.txt`, `submissions/` folder; rewritten introduction, data types, basic statistics and duplicate checks, preprocessing details, conclusion and next steps; notebook run with final outputs |
| 2026-11-13 | Ezgi Nur Temür | Notebook fixes (data download, conclusion), Kaggle submission section, README update |
