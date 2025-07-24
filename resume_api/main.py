import joblib
import __main__
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd

# --- Ensure custom classes available for unpickling ---
from talent_feature_engineer import TalentScoringFeatureEngineer
from model_response_parser import ModelResponseParser
__main__.TalentScoringFeatureEngineer = TalentScoringFeatureEngineer
__main__.ModelResponseParser = ModelResponseParser

# --- Load persisted artifacts at startup ---
# 1. Dataframes
resume_df = pd.read_csv("Challenge-Data/processed_resumes.csv")
jd_df     = pd.read_csv("Challenge-Data/processed_jds.csv")

# 2. Similarity matrices
sim_tfidf       = np.load("Challenge-Data/similarity_matrix_tfidf.npy")
sim_word2vec    = np.load("Challenge-Data/similarity_matrix_word2vec.npy")
sim_transformer = np.load("Challenge-Data/similarity_matrix_transformer.npy")
sim_matrices    = {
    "TF-IDF": sim_tfidf,
    "Word2Vec": sim_word2vec,
    "all-MiniLM-L6-v2": sim_transformer
}

# 3. Models & utilities (load with joblib)
model             = joblib.load("Challenge-Data/talent_model_linear_regression.pkl")
scaler            = joblib.load("Challenge-Data/talent_scaler.pkl")
feature_engineer  = joblib.load("Challenge-Data/talent_feature_engineer.pkl")
parser            = joblib.load("Challenge-Data/model_response_parser.pkl")

# --- Scoring helper function ---
def score_pair(resume_idx: int, jd_idx: int) -> float:
    # Build features for this pair
    features = feature_engineer.create_comprehensive_features(
        resume_idx, jd_idx,
        resume_df, jd_df,
        sim_matrices
    )
    feat_df = pd.DataFrame([features])

    # Scale and predict
    X = scaler.transform(feat_df)
    raw = model.predict(X)[0]

    # Clip to [0, 100]
    return float(max(0, min(100, raw)))

# --- FastAPI app and endpoints ---
app = FastAPI()

@app.get("/")
def root():
    return {"message": "API up & running – try /docs for the Swagger UI"}


@app.get("/jobs/{job_id}/candidates")
def get_top_candidates(job_id: int, top_k: int = 5):
    if job_id < 0 or job_id >= len(jd_df):
        raise HTTPException(status_code=404, detail="Job ID out of range")
    scores = [
        {"resume_id": ridx, "score": score_pair(ridx, job_id)}
        for ridx in range(len(resume_df))
    ]
    best = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
    return best

@app.get("/resumes/{resume_id}/jobs")
def get_top_jobs(resume_id: int, top_k: int = 5):
    if resume_id < 0 or resume_id >= len(resume_df):
        raise HTTPException(status_code=404, detail="Resume ID out of range")
    scores = [
        {"job_id": jidx, "score": score_pair(resume_id, jidx)}
        for jidx in range(len(jd_df))
    ]
    best = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
    return best

# To run:
# uvicorn main:app --reload
