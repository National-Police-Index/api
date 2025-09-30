use pyo3::prelude::*;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    pub jaro_winkler: f64,
    pub levenshtein_norm: f64,
    pub length_ratio: f64,
}

/// Call Python's jellyfish.jaro_winkler_similarity function
fn python_jaro_winkler_similarity(s1: &str, s2: &str) -> PyResult<f64> {
    Python::with_gil(|py| {
        let jellyfish = py.import("jellyfish")?;
        let result: f64 = jellyfish
            .getattr("jaro_winkler_similarity")?
            .call1((s1, s2))?
            .extract()?;
        Ok(result)
    })
}

/// Call Python's jellyfish.levenshtein_distance function
fn python_levenshtein_distance(s1: &str, s2: &str) -> PyResult<i32> {
    Python::with_gil(|py| {
        let jellyfish = py.import("jellyfish")?;
        let result: i32 = jellyfish
            .getattr("levenshtein_distance")?
            .call1((s1, s2))?
            .extract()?;
        Ok(result)
    })
}

/// Calculate comprehensive string similarity metrics between two optional strings
/// Using Python jellyfish library for exact compatibility
pub fn calculate_string_similarity(str1: Option<&str>, str2: Option<&str>) -> Result<SimilarityMetrics> {
    // Handle None/empty cases exactly like Python pd.isna() checks
    match (str1, str2) {
        (Some(s1), Some(s2)) if !s1.trim().is_empty() && !s2.trim().is_empty() => {
            // ONLY lowercase - exactly matching Python logic
            let s1_clean = s1.to_lowercase();
            let s2_clean = s2.to_lowercase();
            
            let max_len = s1_clean.len().max(s2_clean.len()) as f64;
            let min_len = s1_clean.len().min(s2_clean.len()) as f64;
            
            // Handle empty strings after cleaning
            if max_len == 0.0 {
                return Ok(SimilarityMetrics {
                    jaro_winkler: 0.0,
                    levenshtein_norm: 1.0,  // Python returns 1 for max distance
                    length_ratio: 1.0,     // Python returns 1 when max_len == 0
                });
            }
            
            // Call Python jellyfish functions for exact compatibility
            let jw_score = python_jaro_winkler_similarity(&s1_clean, &s2_clean)
                .map_err(|e| anyhow::anyhow!("Python jaro_winkler failed: {}", e))?;
            
            let lev_distance = python_levenshtein_distance(&s1_clean, &s2_clean)
                .map_err(|e| anyhow::anyhow!("Python levenshtein failed: {}", e))?;
            
            // Calculate normalized levenshtein exactly like Python
            let lev_score = 1.0 - (lev_distance as f64 / max_len);
            let len_ratio = min_len / max_len;
            
            Ok(SimilarityMetrics {
                jaro_winkler: jw_score,
                levenshtein_norm: lev_score,
                length_ratio: len_ratio,
            })
        }
        (Some(s1), Some(s2)) if s1.trim().is_empty() && s2.trim().is_empty() => {
            // Both empty strings - matches Python behavior exactly
            Ok(SimilarityMetrics {
                jaro_winkler: 0.0,  // Python returns 0 for empty strings
                levenshtein_norm: 1.0,  // Python returns 1 (max distance)
                length_ratio: 1.0,  // Python returns 1 when max_len == 0
            })
        }
        _ => {
            // One or both are None/empty - return Python's default values
            Ok(SimilarityMetrics {
                jaro_winkler: 0.0,
                levenshtein_norm: 1.0,  // Normalized to [0,1], 1 means maximum distance
                length_ratio: 0.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exact_match() {
        let result = calculate_string_similarity(Some("MEDINA"), Some("MEDINA")).unwrap();
        println!("MEDINA vs MEDINA: JW={}, LEV={}, LEN={}", 
                 result.jaro_winkler, result.levenshtein_norm, result.length_ratio);
        assert!((result.jaro_winkler - 1.0).abs() < 0.001);
        assert!((result.levenshtein_norm - 1.0).abs() < 0.001);
        assert!((result.length_ratio - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_medina_vs_mendez() {
        let result = calculate_string_similarity(Some("MEDINA"), Some("MENDEZ")).unwrap();
        println!("MEDINA vs MENDEZ: JW={:.4}, LEV={:.4}, LEN={:.4}", 
                 result.jaro_winkler, result.levenshtein_norm, result.length_ratio);
        // Should match Python values exactly
        // Python: JW=0.6944, LEV=0.3333, LEN=1.0
        assert!((result.jaro_winkler - 0.6944).abs() < 0.001);
        assert!((result.levenshtein_norm - 0.3333).abs() < 0.01);
        assert!((result.length_ratio - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_case_insensitive() {
        let result1 = calculate_string_similarity(Some("MEDINA"), Some("medina")).unwrap();
        let result2 = calculate_string_similarity(Some("Medina"), Some("MEDINA")).unwrap();
        assert!((result1.jaro_winkler - 1.0).abs() < 0.001);
        assert!((result2.jaro_winkler - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_python_integration() {
        // Test that Python calls work
        assert!(python_jaro_winkler_similarity("test", "test").is_ok());
        assert!(python_levenshtein_distance("test", "test").is_ok());
        
        // Test exact values
        let jw = python_jaro_winkler_similarity("MEDINA", "MENDEZ").unwrap();
        let lev = python_levenshtein_distance("MEDINA", "MENDEZ").unwrap();
        
        println!("Python jellyfish direct calls:");
        println!("  MEDINA vs MENDEZ Jaro-Winkler: {:.4}", jw);
        println!("  MEDINA vs MENDEZ Levenshtein distance: {}", lev);
        
        // These should match Python exactly
        assert!((jw - 0.6944).abs() < 0.001);
        assert_eq!(lev, 4); // Levenshtein distance should be 4
    }
}