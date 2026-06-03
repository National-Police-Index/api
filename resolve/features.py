"""Feature engineering for mention<->candidate scoring.

Behavior-preserving port of legacy resolve/src/features.py. Two changes only:
  - scaler path is resolved relative to this package (not the cwd)
  - the SentenceTransformer import is lazy (keeps importing this module light)

Features MUST match the training set used for the pickled model in resolve/models/.
"""
from __future__ import annotations

import logging
import os
import pickle

import jellyfish
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SCALER_PATH = os.path.join(_MODELS_DIR, "features_scaler.pkl")

with open(SCALER_PATH, "rb") as f:
    TRAINED_SCALER = pickle.load(f)

sentence_transformer_model = None


def _load_st_model():
    global sentence_transformer_model
    if sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return sentence_transformer_model


def calculate_string_similarity(str1, str2):
    if pd.isna(str1) or pd.isna(str2):
        return {"jaro_winkler": 0, "levenshtein_norm": 1, "length_ratio": 0}

    str1 = str(str1).lower()
    str2 = str(str2).lower()
    max_len = max(len(str1), len(str2))
    min_len = min(len(str1), len(str2))
    if max_len == 0:
        return {"jaro_winkler": 0, "levenshtein_norm": 1, "length_ratio": 1}

    return {
        "jaro_winkler": jellyfish.jaro_winkler_similarity(str1, str2),
        "levenshtein_norm": 1 - (jellyfish.levenshtein_distance(str1, str2) / max_len),
        "length_ratio": min_len / max_len if max_len > 0 else 1,
    }


def calculate_embedding_similarity(text1, text2, model):
    if pd.isna(text1) or pd.isna(text2):
        return 0
    e1 = model.encode([str(text1).lower()], convert_to_tensor=True).cpu().numpy()
    e2 = model.encode([str(text2).lower()], convert_to_tensor=True).cpu().numpy()
    return cosine_similarity(e1, e2)[0][0]


def engineer_name_features(df):
    model = _load_st_model()
    features = {}

    name_columns = {
        "first": ("mention_first_name", "post_first_name"),
        "middle": ("mention_middle_name", "post_middle_name"),
        "last": ("mention_last_name", "post_last_name"),
        "suffix": ("mention_suffix", "post_suffix"),
    }

    for name_part, (mention_col, post_col) in name_columns.items():
        features[f"{name_part}_name_jaro"] = []
        features[f"{name_part}_name_levenshtein"] = []
        features[f"{name_part}_name_length_ratio"] = []

        for _, row in df.iterrows():
            if name_part == "middle" and (pd.isna(row[mention_col]) or pd.isna(row[post_col])):
                features[f"{name_part}_name_jaro"].append(0.5)
                features[f"{name_part}_name_levenshtein"].append(0.5)
                features[f"{name_part}_name_length_ratio"].append(0.5)
            else:
                sims = calculate_string_similarity(row[mention_col], row[post_col])
                features[f"{name_part}_name_jaro"].append(sims["jaro_winkler"])
                features[f"{name_part}_name_levenshtein"].append(sims["levenshtein_norm"])
                features[f"{name_part}_name_length_ratio"].append(sims["length_ratio"])

        features[f"{name_part}_name_embedding"] = []
        for _, row in df.iterrows():
            if name_part == "middle" and (pd.isna(row[mention_col]) or pd.isna(row[post_col])):
                features[f"{name_part}_name_embedding"].append(0.5)
            else:
                features[f"{name_part}_name_embedding"].append(
                    calculate_embedding_similarity(row[mention_col], row[post_col], model)
                )

    mention_full_names, post_full_names = [], []
    for _, row in df.iterrows():
        mention_parts = [str(row[c]) for c in
                         ("mention_first_name", "mention_middle_name", "mention_last_name", "mention_suffix")
                         if not pd.isna(row[c])]
        mention_full_names.append(" ".join(mention_parts))
        post_parts = [str(row[c]) for c in
                      ("post_first_name", "post_middle_name", "post_last_name", "post_suffix")
                      if not pd.isna(row[c])]
        post_full_names.append(" ".join(post_parts))

    df["mention_full_name"] = mention_full_names
    df["post_full_name"] = post_full_names

    full_sims = [calculate_string_similarity(m, p)
                 for m, p in zip(mention_full_names, post_full_names)]
    features.update({
        "full_name_jaro": [x["jaro_winkler"] for x in full_sims],
        "full_name_levenshtein": [x["levenshtein_norm"] for x in full_sims],
        "full_name_length_ratio": [x["length_ratio"] for x in full_sims],
    })
    features["full_name_embedding"] = [
        calculate_embedding_similarity(m, p, model)
        for m, p in zip(mention_full_names, post_full_names)
    ]

    for name, values in features.items():
        df[name] = values
    return df


def normalize_features(df):
    features_to_scale = [
        c for c in df.columns
        if any(x in c for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]
    scaled = TRAINED_SCALER.transform(df[features_to_scale])
    df[features_to_scale] = pd.DataFrame(scaled, columns=features_to_scale, index=df.index)
    return df


def featurize(candidates):
    featured_df = engineer_name_features(candidates.copy())
    if featured_df.empty:
        return featured_df
    features_to_scale = [
        c for c in featured_df.columns
        if any(x in c for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]
    if featured_df[features_to_scale].empty:
        return featured_df
    return featured_df
