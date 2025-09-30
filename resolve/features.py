import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import jellyfish
import logging

logger = logging.getLogger("idonea." + __name__)

sentence_transformer_model = None


def calculate_string_similarity(str1, str2):
    """Calculate various string similarity metrics."""
    if pd.isna(str1) or pd.isna(str2):
        return {
            "jaro_winkler": 0,
            "levenshtein_norm": 1,  # Normalized to [0,1], 1 means maximum distance
            "length_ratio": 0,
        }

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
    """Calculate cosine similarity between embeddings of two texts."""
    if pd.isna(text1) or pd.isna(text2):
        return 0

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    embedding1 = model.encode([text1], convert_to_tensor=True)
    embedding2 = model.encode([text2], convert_to_tensor=True)

    embedding1_np = embedding1.cpu().numpy()
    embedding2_np = embedding2.cpu().numpy()

    similarity = cosine_similarity(embedding1_np, embedding2_np)[0][0]

    return similarity


def engineer_name_features(df):
    """Generate both string-based and embedding-based features for names."""
    logger.debug("Loading sentence transformer model...")
    global sentence_transformer_model
    if sentence_transformer_model is None:
        sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = sentence_transformer_model
    else:
        model = sentence_transformer_model

    features = {}

    # Define column mappings
    name_columns = {
        "first": ("mention_first_name", "post_first_name"),
        "middle": ("mention_middle_name", "post_middle_name"),
        "last": ("mention_last_name", "post_last_name"),
        "suffix": ("mention_suffix", "post_suffix"),
    }

    for name_part, (mention_col, post_col) in name_columns.items():
        logger.debug(f"Processing {name_part} names...")

        # Only compute similarities if both values are present
        features[f"{name_part}_name_jaro"] = []
        features[f"{name_part}_name_levenshtein"] = []
        features[f"{name_part}_name_length_ratio"] = []

        for _, row in df.iterrows():
            similarities = calculate_string_similarity(row[mention_col], row[post_col])

            print(f"Similarities {similarities}")
            features[f"{name_part}_name_jaro"].append(similarities["jaro_winkler"])
            features[f"{name_part}_name_levenshtein"].append(
                similarities["levenshtein_norm"]
            )
            features[f"{name_part}_name_length_ratio"].append(
                similarities["length_ratio"]
            )

        # Embedding-based similarity
        features[f"{name_part}_name_embedding"] = []
        for _, row in df.iterrows():
            sim = calculate_embedding_similarity(row[mention_col], row[post_col], model)
            features[f"{name_part}_name_embedding"].append(sim)

    print("Processing full names...")
    mention_full_names = []
    post_full_names = []

    for _, row in df.iterrows():
        # Create mention full name
        mention_parts = []
        if not pd.isna(row["mention_first_name"]):
            mention_parts.append(str(row["mention_first_name"]))
        if not pd.isna(row["mention_middle_name"]):
            mention_parts.append(str(row["mention_middle_name"]))
        if not pd.isna(row["mention_last_name"]):
            mention_parts.append(str(row["mention_last_name"]))
        if not pd.isna(row["mention_suffix"]):
            mention_parts.append(str(row["mention_suffix"]))
        mention_full_names.append(" ".join(mention_parts))

        # Create post full name
        post_parts = []
        if not pd.isna(row["post_first_name"]):
            post_parts.append(str(row["post_first_name"]))
        if not pd.isna(row["post_middle_name"]):
            post_parts.append(str(row["post_middle_name"]))
        if not pd.isna(row["post_last_name"]):
            post_parts.append(str(row["post_last_name"]))
        if not pd.isna(row["post_suffix"]):
            post_parts.append(str(row["post_suffix"]))
        post_full_names.append(" ".join(post_parts))

    df["mention_full_name"] = mention_full_names
    df["post_full_name"] = post_full_names

    # Calculate full name similarities
    full_name_similarities = []
    for mention_full, post_full in zip(mention_full_names, post_full_names):
        full_name_similarities.append(
            calculate_string_similarity(mention_full, post_full)
        )

    # Update features with full name metrics
    features.update(
        {
            "full_name_jaro": [x["jaro_winkler"] for x in full_name_similarities],
            "full_name_levenshtein": [
                x["levenshtein_norm"] for x in full_name_similarities
            ],
            "full_name_length_ratio": [
                x["length_ratio"] for x in full_name_similarities
            ],
        }
    )

    # Add embedding similarity for full names
    features["full_name_embedding"] = []
    for mention_full, post_full in zip(mention_full_names, post_full_names):
        sim = calculate_embedding_similarity(mention_full, post_full, model)
        features["full_name_embedding"].append(sim)

    # Add computed features to dataframe
    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

        # Add debug prints before return
    print("\nRaw feature values for first row:")
    for feature_name, feature_values in features.items():
        if len(feature_values) > 0:
            print(f"{feature_name}: {feature_values[0]}")

    return df


def normalize_features(df):
    """Normalize numerical features to [0,1] range."""
    scaler = MinMaxScaler()

    # Get all features that need scaling
    features_to_scale = [
        col
        for col in df.columns
        if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]

    print("\nBefore normalization (first row):")
    for col in features_to_scale:
        print(f"{col}: {df[col].iloc[0]}")

    # Apply scaling to all features together, not grouped by metric type
    if features_to_scale: 
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    print("\nAfter normalization (first row):")
    for col in features_to_scale:
        print(f"{col}: {df[col].iloc[0]}")

    return df


def featurize(candidates):
    logger.debug("Generating features...")
    print(f"Input candidates shape: {candidates.shape}")
    print(f"First few rows of input:\n{candidates.head()}")

    featured_df = engineer_name_features(candidates.copy())
    print(f"After engineering features shape: {featured_df.shape}")

    features_to_scale = [
        col
        for col in featured_df.columns
        if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]
    print(f"Features to scale: {features_to_scale}")
    print(f"Shape of features to scale: {featured_df[features_to_scale].shape}")
    print(
        f"Any null values in features?: {featured_df[features_to_scale].isnull().any().any()}"
    )

    if featured_df.empty:
        print("DataFrame is empty after feature engineering!")
        return featured_df

    df_to_normalize = featured_df[features_to_scale]
    if df_to_normalize.empty:
        print("No features to normalize!")
        return featured_df

    # # Only normalize if we have data
    # if len(df_to_normalize) > 0:
    #     featured_df = normalize_features(featured_df)

    # print("\nFinal feature order:")
    # print(features_to_scale)

    return featured_df
