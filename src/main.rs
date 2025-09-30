use anyhow::{Result, Context};
use polars::prelude::*;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel,
};
use std::sync::Arc;
use pyo3::prelude::*;

mod string_similarity;

use string_similarity::{calculate_string_similarity, SimilarityMetrics};

pub struct FeatureEngineer {
    embedding_model: Arc<SentenceEmbeddingsModel>,
}

impl FeatureEngineer {
    pub fn new() -> Result<Self> {
        // Initialize Python interpreter for jellyfish calls
        pyo3::prepare_freethreaded_python();
        
        // Load the sentence embedding model
        let embedding_model = Arc::new(
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()
                .context("Failed to load sentence embedding model")?
        );
        
        Ok(Self {
            embedding_model,
        })
    }
    
    pub fn featurize(&self, df: DataFrame) -> Result<DataFrame> {
        let featured_df = self.engineer_name_features(df)?;
        Ok(featured_df)
    }
    
    fn engineer_name_features(&self, mut df: DataFrame) -> Result<DataFrame> {
        // Define name column mappings
        let name_columns = vec![
            ("first", "mention_first_name", "post_first_name"),
            ("middle", "mention_middle_name", "post_middle_name"),
            ("last", "mention_last_name", "post_last_name"),
            ("suffix", "mention_suffix", "post_suffix"),
        ];
        
        // Process each name component
        for (name_part, mention_col, post_col) in &name_columns {
            // Get the columns as string vectors
            let mention_values = self.extract_string_column(&df, mention_col)?;
            let post_values = self.extract_string_column(&df, post_col)?;
            
            // Calculate string similarities and embeddings
            let mut jaro_values = Vec::new();
            let mut levenshtein_values = Vec::new();
            let mut length_ratio_values = Vec::new();
            let mut embedding_values = Vec::new();
            
            for i in 0..mention_values.len() {
                // String similarities using Python jellyfish
                let similarities = calculate_string_similarity(
                    mention_values[i].as_deref(), 
                    post_values[i].as_deref()
                )?;
                
                jaro_values.push(similarities.jaro_winkler);
                levenshtein_values.push(similarities.levenshtein_norm);
                length_ratio_values.push(similarities.length_ratio);
                
                // Embedding similarity using Rust
                let embedding_sim = self.calculate_embedding_similarity_single(
                    &mention_values[i], 
                    &post_values[i]
                )?;
                embedding_values.push(embedding_sim);
            }
            
            // Add features to dataframe
            df = df.lazy()
                .with_column(Series::new(&format!("{}_name_jaro", name_part), jaro_values).lit())
                .with_column(Series::new(&format!("{}_name_levenshtein", name_part), levenshtein_values).lit())
                .with_column(Series::new(&format!("{}_name_length_ratio", name_part), length_ratio_values).lit())
                .with_column(Series::new(&format!("{}_name_embedding", name_part), embedding_values).lit())
                .collect()?;
        }
        
        // Process full names
        let (mention_full_names, post_full_names) = self.create_full_names(&df)?;
        
        // Add full name columns to dataframe
        df = df.lazy()
            .with_column(Series::new("mention_full_name", &mention_full_names).lit())
            .with_column(Series::new("post_full_name", &post_full_names).lit())
            .collect()?;
        
        // Calculate full name similarities
        let mut full_jaro = Vec::new();
        let mut full_levenshtein = Vec::new();
        let mut full_length_ratio = Vec::new();
        let mut full_embedding = Vec::new();
        
        for i in 0..mention_full_names.len() {
            // String similarities using Python jellyfish
            let similarities = calculate_string_similarity(
                Some(&mention_full_names[i]), 
                Some(&post_full_names[i])
            )?;
            
            full_jaro.push(similarities.jaro_winkler);
            full_levenshtein.push(similarities.levenshtein_norm);
            full_length_ratio.push(similarities.length_ratio);
            
            // Embedding similarity using Rust
            let embedding_sim = self.calculate_embedding_similarity_single(
                &Some(mention_full_names[i].clone()), 
                &Some(post_full_names[i].clone())
            )?;
            full_embedding.push(embedding_sim);
        }
        
        df = df.lazy()
            .with_column(Series::new("full_name_jaro", full_jaro).lit())
            .with_column(Series::new("full_name_levenshtein", full_levenshtein).lit())
            .with_column(Series::new("full_name_length_ratio", full_length_ratio).lit())
            .with_column(Series::new("full_name_embedding", full_embedding).lit())
            .collect()?;
        
        Ok(df)
    }
    
    fn extract_string_column(&self, df: &DataFrame, col_name: &str) -> Result<Vec<Option<String>>> {
        match df.column(col_name) {
            Ok(series) => {
                let values: Vec<Option<String>> = series
                    .iter()
                    .map(|val| {
                        match val {
                            AnyValue::Utf8(s) if !s.is_empty() => Some(s.to_string()),
                            AnyValue::Null => None,
                            _ => None,
                        }
                    })
                    .collect();
                Ok(values)
            }
            Err(_) => {
                Ok(vec![None; df.height()])
            }
        }
    }
    
    fn create_full_names(&self, df: &DataFrame) -> Result<(Vec<String>, Vec<String>)> {
        let mention_first = self.extract_string_column(df, "mention_first_name")?;
        let mention_middle = self.extract_string_column(df, "mention_middle_name")?;
        let mention_last = self.extract_string_column(df, "mention_last_name")?;
        let mention_suffix = self.extract_string_column(df, "mention_suffix")?;
        
        let post_first = self.extract_string_column(df, "post_first_name")?;
        let post_middle = self.extract_string_column(df, "post_middle_name")?;
        let post_last = self.extract_string_column(df, "post_last_name")?;
        let post_suffix = self.extract_string_column(df, "post_suffix")?;
        
        let mention_full_names: Vec<String> = (0..df.height())
            .map(|i| {
                let mut parts = Vec::new();
                
                if let Some(ref name) = mention_first[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = mention_middle[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = mention_last[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = mention_suffix[i] {
                    parts.push(name.as_str());
                }
                
                parts.join(" ")
            })
            .collect();
        
        let post_full_names: Vec<String> = (0..df.height())
            .map(|i| {
                let mut parts = Vec::new();
                
                if let Some(ref name) = post_first[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = post_middle[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = post_last[i] {
                    parts.push(name.as_str());
                }
                if let Some(ref name) = post_suffix[i] {
                    parts.push(name.as_str());
                }
                
                parts.join(" ")
            })
            .collect();
        
        Ok((mention_full_names, post_full_names))
    }
    
    fn calculate_embedding_similarity_single(
        &self, 
        text1: &Option<String>, 
        text2: &Option<String>
    ) -> Result<f64> {
        // Handle None values exactly like Python pd.isna() check
        let t1 = match text1 {
            Some(s) if !s.trim().is_empty() => s.to_lowercase(),
            _ => return Ok(0.0),
        };
        
        let t2 = match text2 {
            Some(s) if !s.trim().is_empty() => s.to_lowercase(),
            _ => return Ok(0.0),
        };
        
        // Calculate embeddings
        let embeddings1 = self.embedding_model.encode(&[t1])?;
        let embeddings2 = self.embedding_model.encode(&[t2])?;
        
        // Calculate cosine similarity
        let similarity = cosine_similarity(&embeddings1[0], &embeddings2[0]);
        
        Ok(similarity as f64)
    }
}

// Helper function to calculate cosine similarity
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }
    
    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|a| a * a).sum::<f32>().sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_csv> <output_csv>", args[0]);
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = &args[2];
    
    println!("Loading data from {}...", input_path);
    let df = CsvReader::from_path(input_path)?
        .has_header(true)
        .finish()?;
    
    println!("Loaded {} rows", df.height());
    
    // Initialize feature engineer
    let feature_engineer = FeatureEngineer::new()?;
    
    // Process features
    let featured_df = feature_engineer.featurize(df)?;
    
    // Save to CSV
    let mut file = std::fs::File::create(output_path)?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut featured_df.clone())?;
    
    println!("Feature engineering complete! Generated {} features for {} rows", 
             featured_df.width(), featured_df.height());
    
    Ok(())
}