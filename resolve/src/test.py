#!/usr/bin/env python3
"""
Debug script to test PyO3-integrated Rust feature engineering
"""

import pandas as pd
import datetime
from match import generate_candidates, PostMatcher
from features_rust import featurize as rust_featurize
from features import featurize as python_featurize
from models.src import OfficerMention

def test_pyo3_integration():
    """Test if PyO3-integrated Rust produces identical results to Python"""
    
    # Create the exact mention that was failing
    mention = OfficerMention(
        mention_uid="0da8b4868c04b693d3efa4b08a156870",
        mention_agency_type="POLICE",
        mention_incident_date=datetime.date(2018, 1, 1),
        mention_first_name="ROBERT",
        mention_middle_name="",
        mention_last_name="MEDINA",
        mention_agency="Los Angeles Police Department",
        state=None
    )
    
    print("=== TESTING PYO3-INTEGRATED RUST FEATURE ENGINEERING ===")
    
    # Get candidates
    candidates = generate_candidates(mention)
    print(f"Total candidates: {len(candidates)}")
    
    # Focus on the problematic cases
    exact_medina = candidates[
        (candidates['post_first_name'] == 'ROBERT') & 
        (candidates['post_last_name'] == 'MEDINA')
    ].copy()
    
    mendez_candidate = candidates[
        (candidates['post_first_name'] == 'ROBERT') & 
        (candidates['post_last_name'] == 'MENDEZ')
    ].copy()
    
    test_candidates = pd.concat([exact_medina, mendez_candidate])
    if len(test_candidates) == 0:
        print("No test candidates found!")
        return
    
    print(f"\nTesting with {len(test_candidates)} candidates:")
    for idx, row in test_candidates.iterrows():
        print(f"  - {row['post_first_name']} {row['post_last_name']} ({row['post_agency_name']})")
    
    # Test Pure Python Features
    print(f"\n{'='*60}")
    print("TESTING PURE PYTHON FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    try:
        python_features = python_featurize(test_candidates.copy())
        print("✅ Python feature engineering completed")
        
        print("\nPython key feature values:")
        key_features = ['last_name_jaro', 'last_name_levenshtein', 'full_name_jaro']
        for idx, row in test_candidates.iterrows():
            candidate_name = f"{row['post_first_name']} {row['post_last_name']}"
            row_idx = test_candidates.index.get_loc(idx)
            print(f"\n{candidate_name}:")
            for feat in key_features:
                if feat in python_features.columns:
                    val = python_features[feat].iloc[row_idx]
                    print(f"  {feat}: {val:.6f}")
        
    except Exception as e:
        print(f"❌ Python feature engineering failed: {e}")
        python_features = None
    
    # Test PyO3-Integrated Rust Features
    print(f"\n{'='*60}")
    print("TESTING PYO3-INTEGRATED RUST FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    try:
        rust_features = rust_featurize(test_candidates.copy())
        print("✅ PyO3-Rust feature engineering completed")
        
        print("\nRust key feature values:")
        for idx, row in test_candidates.iterrows():
            candidate_name = f"{row['post_first_name']} {row['post_last_name']}"
            row_idx = test_candidates.index.get_loc(idx)
            print(f"\n{candidate_name}:")
            for feat in key_features:
                if feat in rust_features.columns:
                    val = rust_features[feat].iloc[row_idx]
                    print(f"  {feat}: {val:.6f}")
        
    except Exception as e:
        print(f"❌ PyO3-Rust feature engineering failed: {e}")
        rust_features = None
    
    # Compare Results
    if python_features is not None and rust_features is not None:
        print(f"\n{'='*60}")
        print("DETAILED FEATURE COMPARISON")
        print(f"{'='*60}")
        
        all_feature_cols = ['first_name_jaro', 'first_name_levenshtein', 'first_name_length_ratio',
                           'last_name_jaro', 'last_name_levenshtein', 'last_name_length_ratio',
                           'full_name_jaro', 'full_name_levenshtein', 'full_name_length_ratio']
        
        perfect_matches = 0
        total_comparisons = 0
        
        for idx, row in test_candidates.iterrows():
            candidate_name = f"{row['post_first_name']} {row['post_last_name']}"
            row_idx = test_candidates.index.get_loc(idx)
            
            print(f"\n{candidate_name} (comparing {row['mention_last_name']} vs {row['post_last_name']}):")
            
            row_perfect = True
            for feat in all_feature_cols:
                if feat in python_features.columns and feat in rust_features.columns:
                    python_val = python_features[feat].iloc[row_idx]
                    rust_val = rust_features[feat].iloc[row_idx]
                    diff = abs(python_val - rust_val)
                    
                    # Determine status
                    if diff < 0.0001:
                        status = "✅ PERFECT"
                        perfect_matches += 1
                    elif diff < 0.001:
                        status = "✅ EXCELLENT"
                    elif diff < 0.01:
                        status = "⚠️  MINOR DIFF"
                        row_perfect = False
                    else:
                        status = "❌ MAJOR DIFF"
                        row_perfect = False
                    
                    total_comparisons += 1
                    print(f"  {feat:25}: Python={python_val:.6f}, Rust={rust_val:.6f}, Diff={diff:.6f} {status}")
            
            if row_perfect:
                print(f"  🎯 {candidate_name}: ALL FEATURES MATCH PERFECTLY!")
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Perfect matches: {perfect_matches}/{total_comparisons} ({perfect_matches/total_comparisons*100:.1f}%)")
        
        if perfect_matches == total_comparisons:
            print("🎉 SUCCESS: All features match perfectly between Python and PyO3-Rust!")
        elif perfect_matches / total_comparisons > 0.9:
            print("✅ VERY GOOD: Most features match, minor differences only")
        else:
            print("⚠️  ISSUES: Significant differences found, needs investigation")
    
    # Test Model Predictions with PyO3-Rust Features
    if rust_features is not None:
        print(f"\n{'='*60}")
        print("TESTING MODEL PREDICTIONS WITH PYO3-RUST FEATURES")
        print(f"{'='*60}")
        
        matcher = PostMatcher()
        model = matcher._xgboost_model()
        
        feature_cols = [col for col in rust_features.columns 
                       if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])]
        model_features = [c for c in feature_cols if c in model.feature_names_in_]
        
        probabilities = model.predict_proba(rust_features[model_features])
        rust_features['match_probability'] = probabilities[:, 1]
        
        rust_features_sorted = rust_features.sort_values('match_probability', ascending=False)
        
        print(f"\nModel predictions using PyO3-Rust features:")
        for idx, row in rust_features_sorted.iterrows():
            print(f"  {row['post_first_name']} {row['post_last_name']:8} -> {row['match_probability']:.6f}")
        
        # Check if MEDINA beats MENDEZ
        medina_probs = rust_features_sorted[rust_features_sorted['post_last_name'] == 'MEDINA']['match_probability'].values
        mendez_probs = rust_features_sorted[rust_features_sorted['post_last_name'] == 'MENDEZ']['match_probability'].values
        
        if len(medina_probs) > 0 and len(mendez_probs) > 0:
            if medina_probs.max() > mendez_probs.max():
                print("\n🎉 SUCCESS: MEDINA has higher probability than MENDEZ!")
                print(f"   Best MEDINA: {medina_probs.max():.6f}")
                print(f"   Best MENDEZ: {mendez_probs.max():.6f}")
            else:
                print("\n❌ STILL BROKEN: MENDEZ has higher probability than MEDINA")
                print(f"   Best MEDINA: {medina_probs.max():.6f}")
                print(f"   Best MENDEZ: {mendez_probs.max():.6f}")

def quick_similarity_test():
    """Quick test of individual string similarities"""
    print(f"\n{'='*60}")
    print("QUICK STRING SIMILARITY TEST")
    print(f"{'='*60}")
    
    test_pairs = [
        ("MEDINA", "MEDINA"),
        ("MEDINA", "MENDEZ"),
        ("ROBERT", "ROBERT"),
        ("ROBERT MEDINA", "ROBERT MEDINA"),
        ("ROBERT MEDINA", "ROBERT MENDEZ"),
    ]
    
    # Import Python jellyfish for comparison
    import jellyfish
    
    print("Testing string similarities:")
    for s1, s2 in test_pairs:
        python_jw = jellyfish.jaro_winkler_similarity(s1.lower(), s2.lower())
        python_lev_dist = jellyfish.levenshtein_distance(s1.lower(), s2.lower())
        python_lev_norm = 1 - (python_lev_dist / max(len(s1), len(s2)))
        
        print(f"\n'{s1}' vs '{s2}':")
        print(f"  Python JW: {python_jw:.6f}")
        print(f"  Python LEV: {python_lev_norm:.6f}")

if __name__ == "__main__":
    quick_similarity_test()
    test_pyo3_integration()