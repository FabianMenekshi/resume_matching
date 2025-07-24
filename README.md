# Talent Scoring & Matching API

A Python‑based project for computing compatibility scores between candidate resumes and job descriptions, training a regression model to predict those scores, and exposing the functionality via a RESTful FastAPI service.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Jupyter Notebook Workflow](#jupyter-notebook-workflow)
   - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
   - [Step 2: Feature Engineering & Model Training](#step-2-feature-engineering--model-training)
   - [Step 3: Scoring & Evaluation](#step-3-scoring--evaluation)
3. [API Service](#api-service)
   - [Architecture](#architecture)
   - [Endpoints](#endpoints)
   - [Usage Examples](#usage-examples)
4. [Technologies & Dependencies](#technologies--dependencies)
5. [File Structure](#file-structure)
6. [Installation & Setup](#installation--setup)
7. [Contributing](#contributing)

---

## Project Overview

This repository provides:

1. A **data science pipeline** (Jupyter notebook) to:

   - Load and preprocess resumes and job descriptions.
   - Engineer features combining skills, experience, and semantic similarities (TF‑IDF, Word2Vec, transformer embeddings).
   - Train and evaluate a regression model (Linear Regression) to predict a compatibility score (0–100).

2. A **FastAPI service** that:

   - Loads the trained model, scaler, feature engineer, and similarity matrices at startup.
   - Exposes two endpoints:
     - `GET /jobs/{job_id}/candidates?top_k=N`: top N resumes for a given job.
     - `GET /resumes/{resume_id}/jobs?top_k=N`: top N jobs for a given resume.

3. All artifacts (processed data, pickled objects, similarity matrices) in `Challenge-Data/`.

---

## Jupyter Notebook Workflow

The notebook (`talent_scoring_pipeline.ipynb`) is organized into three main steps:

### Step 1: Data Preprocessing

- **Load Raw Data**

  - Read raw resume files (e.g., PDFs or text dumps) into a DataFrame.
  - Read raw job descriptions into a second DataFrame.

- **Cleaning & Normalization**

  - Strip HTML tags, lowercase text, remove punctuation.
  - Fill missing values and filter out unusable records.
  - Compute basic statistics: `text_length`, `word_count`.

- **Persist Processed CSVs**

  - Save cleaned DataFrames as `processed_resumes.csv` and `processed_jds.csv` for reproducibility.

### Step 2: Feature Engineering & Model Training

- **Feature Engineering** (in `talent_feature_engineer.py`)

  - **Skill Match Features**: Count and ratio of overlapping skills between resume and job (based on parsed skill lists).
  - **Experience Match Features**: Extract years of experience from text and compare to job requirements.
  - **Semantic Similarity Features**:
    - **TF‑IDF Cosine Similarity** between cleaned resume & job-description vectors.
    - **Word2Vec Similarity**: average of token‑vector similarities.
    - **Transformer Embeddings** (`all‑MiniLM‑L6‑v2`) cosine similarity.

- **Assemble Feature Matrix**

  - Combine engineered features into a single DataFrame.
  - Scale features with a `StandardScaler` (persisted as `talent_scaler.pkl`).

- **Model Training**

  - Split into train/test sets.
  - Train a `LinearRegression` model to predict a 0–100 compatibility score (persisted as `talent_model_linear_regression.pkl`).
  - Evaluate using R² score, error metrics, and manual inspection of top matches.

### Step 3: Scoring & Evaluation

- **Generate Compatibility Scores**

  - For each (resume, job) pair, compute feature vector and run through scaler + model.
  - Clip predictions to `[0, 100]`.

- **Manual Validation**

  - Inspect top‑5 matches for sample jobs and resumes.
  - Confirm that high scores align with expected skill and experience overlaps.

- **Persist Similarity Matrices**

  - Save `similarity_matrix_tfidf.npy`, `similarity_matrix_word2vec.npy`, `similarity_matrix_transformer.npy` for fast lookup in the API.

---

## API

The FastAPI application lives in `main.py`. On startup, it loads:

- **DataFrames**: `processed_resumes.csv`, `processed_jds.csv` via pandas.
- **Similarity matrices**: `.npy` files via NumPy.
- **Model artifacts**: `talent_model_linear_regression.pkl`, `talent_scaler.pkl`, `talent_feature_engineer.pkl`, and (optionally) `model_response_parser.pkl` via joblib.

### Architecture

1. **Feature Engineering**: `TalentScoringFeatureEngineer.create_comprehensive_features(...)` builds the per-pair feature dict.
2. **Prediction**: features → `StandardScaler.transform` → `LinearRegression.predict` → clipped score.
3. **Endpoints**:
   - ``
     - Validates `job_id` range.
     - Scores all resumes → sorts → returns top‑K JSON list of `[{resume_id, score}]`.
   - ``
     - Validates `resume_id` range.
     - Scores all jobs → sorts → returns top‑K JSON list of `[{job_id, score}]`.

### Usage Examples

```bash
# Startup (in project root)
uvicorn main:app --reload

# Swagger UI
ohttp://127.0.0.1:8000/docs

# Fetch top 5 resumes for job #0
ohttp GET "http://127.0.0.1:8000/jobs/0/candidates?top_k=5"

# Fetch top 3 jobs for resume #10
ohttp GET "http://127.0.0.1:8000/resumes/10/jobs?top_k=3"
```

---

## Technologies & Dependencies

- **Language**: Python 3.10+
- **Data Analysis**: pandas, NumPy
- **NLP / Similarity**: scikit‑learn (TF‑IDF), gensim (Word2Vec), sentence-transformers
- **Modeling**: scikit‑learn (LinearRegression, StandardScaler)
- **Serialization**: joblib, pickle
- **Web Framework**: FastAPI, Uvicorn
- **Utilities**: pathlib, logging, re
- **Development**: Jupyter Notebook

Dependencies are pinned in `requirements.txt`.

---

## File Structure

```
├── Challenge-Data/                      # Persisted artifacts
│   ├── processed_resumes.csv
│   ├── processed_jds.csv
│   ├── similarity_matrix_tfidf.npy
│   ├── similarity_matrix_word2vec.npy
│   ├── similarity_matrix_transformer.npy
│   ├── talent_model_linear_regression.pkl
│   ├── talent_scaler.pkl
│   ├── talent_feature_engineer.pkl
│   └── model_response_parser.pkl
│
├── resume_api/                          # API application
│   ├── main.py                          # FastAPI app
│   ├── talent_feature_engineer.py       # Feature engineering class
│   ├── model_response_parser.py         # (Optional) parser utility
│   └── __init__.py                      # Python package
│
├── notebooks/                           # Data science pipeline
│   └── talent_scoring_pipeline.ipynb    # Step-by-step notebook
│
├── requirements.txt
├── README.md                            # ← you are here
└── .gitignore
```

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv        # or python3
   source venv/Scripts/activate   # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Data Artifacts** Ensure `Challenge-Data/` contains:

   - `processed_resumes.csv`, `processed_jds.csv`
   - `.npy` similarity matrices
   - `.pkl` model & utilities

5. **Run the API**

   ```bash
   uvicorn main:app --reload
   ```

   - Browse the interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

6. **Test Endpoints**

   ```bash
   curl "http://127.0.0.1:8000/jobs/0/candidates?top_k=5"
   curl "http://127.0.0.1:8000/resumes/10/jobs?top_k=3"
   ```
