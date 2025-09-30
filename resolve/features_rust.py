"""
Rust Feature Engineering Wrapper
"""

import subprocess
import pandas as pd
import tempfile
import os
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger("idonea." + __name__)

class RustFeatureError(Exception):
    """Custom exception for Rust feature engineering errors"""
    pass

class RustFeatureEngineer:
    """Singleton wrapper for Rust feature engineering binary"""
    
    _instance = None
    _binary_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize and verify the Rust binary path"""
        # Try multiple possible locations for the binary
        possible_paths = [
            Path("./target/release/features"),
            Path("../target/release/features"), 
            Path("target/release/features"),
            Path("./features"),
            Path("./api/target/release/features"),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                self._binary_path = str(path.absolute())
                logger.info(f"Found Rust feature engineering binary at: {self._binary_path}")
                return
        
        raise RustFeatureError(
            f"Rust feature engineering binary not found. Checked paths: {[str(p) for p in possible_paths]}\n"
            "Please ensure the Rust binary is built with: cargo build --release"
        )
    
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features using the Rust pipeline.
        
        Args:
            df: Input DataFrame with mention and post columns
            
        Returns:
            DataFrame with additional feature columns
            
        Raises:
            RustFeatureError: If the Rust binary fails or produces invalid output
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to featurize()")
            return df
        
        logger.debug(f"Featurizing DataFrame with shape: {df.shape}")
        
        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            input_path = input_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Write input DataFrame to CSV
            df.to_csv(input_path, index=False)
            logger.debug(f"Wrote input CSV to: {input_path}")
            
            # Call Rust binary
            logger.debug(f"Calling Rust binary: {self._binary_path}")
            result = subprocess.run([
                self._binary_path,
                input_path,
                output_path
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            # Check for errors
            if result.returncode != 0:
                error_msg = f"Rust feature engineering failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                logger.error(error_msg)
                raise RustFeatureError(error_msg)
            
            # Log Rust output for debugging
            if result.stdout:
                logger.debug(f"Rust binary output:\n{result.stdout}")
            
            # Read the results
            if not os.path.exists(output_path):
                raise RustFeatureError(f"Rust binary did not create output file: {output_path}")
            
            featured_df = pd.read_csv(output_path)
            logger.debug(f"Successfully read featured DataFrame with shape: {featured_df.shape}")
            
            # Validate output
            if featured_df.empty:
                raise RustFeatureError("Rust binary produced empty output")
            
            if len(featured_df) != len(df):
                raise RustFeatureError(
                    f"Row count mismatch: input={len(df)}, output={len(featured_df)}"
                )
            
            # Log feature engineering statistics
            feature_cols = [col for col in featured_df.columns 
                          if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])]
            logger.info(f"Generated {len(feature_cols)} features for {len(featured_df)} rows")
            logger.debug(f"Generated features: {feature_cols}")
            
            return featured_df
            
        except subprocess.TimeoutExpired:
            error_msg = "Rust feature engineering timed out after 5 minutes"
            logger.error(error_msg)
            raise RustFeatureError(error_msg)
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise RustFeatureError(f"Feature engineering failed: {str(e)}")
            
        finally:
            # Cleanup temporary files
            for path in [input_path, output_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                        logger.debug(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {path}: {e}")


# Global instance for efficient reuse
_rust_engineer = None

def featurize(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Drop-in replacement for the Python featurize() function.
    Uses Rust implementation for improved performance.
    
    Args:
        candidates: DataFrame with mention and post candidate pairs
        
    Returns:
        DataFrame with engineered features
    """
    global _rust_engineer
    
    if _rust_engineer is None:
        _rust_engineer = RustFeatureEngineer()
    
    return _rust_engineer.featurize(candidates)


def test_rust_features(test_file: Optional[str] = None):
    """
    Test the Rust feature engineering pipeline.
    
    Args:
        test_file: Optional path to test CSV file
    """
    if test_file and os.path.exists(test_file):
        print(f"Testing with file: {test_file}")
        df = pd.read_csv(test_file)
    else:
        print("Creating test DataFrame...")
        # Create minimal test data
        df = pd.DataFrame({
            'mention_first_name': ['John', 'Jane'],
            'mention_last_name': ['Doe', 'Smith'], 
            'mention_middle_name': ['A', ''],
            'mention_suffix': ['', 'Jr'],
            'post_first_name': ['Jon', 'Jane'],
            'post_last_name': ['Doe', 'Smith'],
            'post_middle_name': ['', ''],
            'post_suffix': ['', 'Jr'],
        })
    
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {list(df.columns)}")
    
    try:
        result = featurize(df)
        print(f"Output shape: {result.shape}")
        
        feature_cols = [col for col in result.columns 
                       if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])]
        print(f"Generated {len(feature_cols)} feature columns:")
        for col in feature_cols:
            print(f"  - {col}")
        
        print("\nFirst row feature values:")
        for col in feature_cols:
            if col in result.columns:
                print(f"  {col}: {result[col].iloc[0]}")
        
        print("\nTest completed successfully!")
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rust feature engineering")
    parser.add_argument("--test", help="Path to test CSV file")
    args = parser.parse_args()
    
    test_rust_features(args.test)