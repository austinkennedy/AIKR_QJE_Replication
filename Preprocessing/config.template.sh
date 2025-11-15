#!/bin/bash
# ============================================================================
# HTRC PREPROCESSING CONFIGURATION
# ============================================================================
# Copy this file to config.sh and customize for your environment:
#   cp config.template.sh config.sh
#   vim config.sh
#
# Then run:
#   python preprocess_htrc.py --config config.sh
#
# Command-line arguments override these settings.
# ============================================================================

# ============================================================================
# REQUIRED PATHS
# ============================================================================

# Input directory containing HTRC Extracted Features files (.json.bz2)
# Example: "/path/to/htrc_extracted_features"
INPUT_DIR=""

# Output directory for cleaned text files (will be created)
# Example: "/path/to/cleaned_text_output"
OUTPUT_DIR=""

# ============================================================================
# DICTIONARY FILES (Required for replication)
# ============================================================================
# These files should be in the reference_data/ directory.
# Default paths assume you're running from the Preprocessing/ directory.
# ============================================================================

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Spelling corrections dictionary
DICT_CORRECTIONS="$SCRIPT_DIR/reference_data/Master_Corrections.csv"

# Modern/Archaic word mappings
DICT_MA="$SCRIPT_DIR/reference_data/MA_Dict_Final.csv"

# World cities for geographic stopword filtering
WORLD_CITIES="$SCRIPT_DIR/reference_data/world_cities.csv"

# ============================================================================
# OPTIONAL SETTINGS
# ============================================================================

# Number of CPU processes for parallel processing
# Leave empty for auto-detect (recommended)
# Example: NUM_PROCESSES="8"
NUM_PROCESSES=""

# Error log file path
# Leave empty to log errors to stderr only
# Example: ERROR_LOG="./preprocessing_errors.log"
ERROR_LOG=""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# After editing this file:
#
#   # Run preprocessing
#   python preprocess_htrc.py --config config.sh
#
#   # Preview without executing
#   python preprocess_htrc.py --config config.sh --dry-run
#
#   # Override output directory
#   python preprocess_htrc.py --config config.sh --output /different/path
#
#   # Specify number of processes
#   python preprocess_htrc.py --config config.sh --num-processes 16
#
# ============================================================================
