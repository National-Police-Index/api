#!/bin/bash

# Build script for Rust feature engineering

set -e

echo "🦀 Building Rust Feature Engineering Pipeline"
echo "============================================="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Rust/Cargo found"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Cargo.toml not found. Please run this script from the project root."
    exit 1
fi

echo "📦 Installing dependencies and building..."

# Build in release mode for maximum performance
cargo build --release

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🎯 Binary location: target/release/features"
    echo ""
    echo "🐍 To use from Python:"
    echo "   from features_rust import featurize"
    echo "   result = featurize(your_dataframe)"
    echo ""
    echo "🚀 To run directly:"
    echo "   ./target/release/features input.csv output.csv"
    echo ""
    echo "🧪 To test:"
    echo "   python features_rust.py --test your_test_file.csv"
else
    echo "❌ Build failed!"
    exit 1
fi