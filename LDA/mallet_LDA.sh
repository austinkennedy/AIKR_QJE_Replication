#!/bin/bash

# ============================================================================
# SLURM CONFIGURATION (For HPC Users)
# ============================================================================
# Customize these headers for your HPC environment:
# - Change --account to your allocation
# - Adjust --mem, --time, --partition for your cluster
# - Remove these headers entirely if running locally (non-SLURM)
# ============================================================================

#SBATCH --time=40:00:00
#SBATCH --qos=mem
#SBATCH --partition=amem
#SBATCH --ntasks=48
#SBATCH --mem=500000
#SBATCH --nodes=1
#SBATCH --job-name=master-mallet
#SBATCH --output=mallet_run_%j.out    # %j = job ID (auto-generated)
#SBATCH --account=ucb593_asc1          # CHANGE THIS to your account

# ============================================================================
# MALLET Topic Modeling - Replication Script
# ============================================================================
# Purpose: Exact replication of topic modeling analysis
#
# WARNING: Model parameters (num-topics, random-seed) are INTENTIONALLY
# HARDCODED for reproducibility. Changing these produces different results
# and breaks replication. Only configure infrastructure parameters (paths).
#
# Version: 2.0
# Last Updated: 2025-10-25
# ============================================================================

# ============================================================================
# MODEL PARAMETERS - DO NOT MODIFY (Ensures Exact Replication)
# ============================================================================
readonly NUM_TOPICS=60                # Number of topics in the model
readonly RANDOM_SEED=1                # RNG seed for reproducibility
readonly OPTIMIZE_INTERVAL=500        # Hyperparameter optimization frequency

# ============================================================================

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ============================================================================
# REQUIRED: Set these paths
INPUT_DIR=""                          # Where your cleaned text files are
OUTPUT_DIR=""                         # Where results will be saved

# OPTIONAL: Leave empty for default
NUM_THREADS=""                        # CPU threads (empty = auto-detect)

# HPC Module loading (ignored if modules don't exist)
MODULE_PURGE=true
MODULE_JAVA="jdk/1.8.0"
# ============================================================================

# Get script directory for finding stoplist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_threads() {
    if [[ -z "$NUM_THREADS" ]]; then
        # Try various methods to detect CPU count
        if command -v nproc &> /dev/null; then
            NUM_THREADS=$(nproc)
        elif [[ -f /proc/cpuinfo ]]; then
            NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)
        elif command -v sysctl &> /dev/null; then
            NUM_THREADS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        else
            NUM_THREADS=4  # Conservative default
        fi
        echo "  ✓ Auto-detected $NUM_THREADS threads"
    else
        echo "  ✓ Using $NUM_THREADS threads (user-specified)"
    fi
}

main() {
    echo "================================================================================"
    echo "MALLET Topic Modeling - Replication Script"
    echo "================================================================================"
    echo ""

    # Module loading (HPC only)
    if $MODULE_PURGE && command -v module &> /dev/null; then
        echo "Loading HPC modules..."
        module purge
        module load $MODULE_JAVA
        echo "  ✓ Modules loaded"
        echo ""
    fi

    # Auto-detect threads
    detect_threads
    echo ""

    # Hardcoded stoplist
    STOPLIST_FILE="$SCRIPT_DIR/words_to_delete.txt"
    STOPLIST_ARG="--stoplist-file \"$STOPLIST_FILE\""

    # Display configuration
    echo ""
    echo "Configuration:"
    echo "  Input Directory:      $INPUT_DIR"
    echo "  Output Directory:     $OUTPUT_DIR"
    echo "  Stoplist:             $STOPLIST_FILE"
    echo "  Threads:              $NUM_THREADS"
    echo "  Topics:               $NUM_TOPICS (fixed)"
    echo "  Random Seed:          $RANDOM_SEED (fixed)"
    echo "  Optimization:         Every $OPTIMIZE_INTERVAL iterations (fixed)"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Step 1: Import documents
    echo "================================================================================"
    echo "Step 1/2: Importing documents..."
    echo "================================================================================"

    mallet import-dir \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR/input.mallet" \
        --keep-sequence \
        $STOPLIST_ARG

    echo "  ✓ Documents imported successfully"
    echo ""

    # Step 2: Train topics
    echo "================================================================================"
    echo "Step 2/2: Training topic model..."
    echo "================================================================================"
    echo "(This may take a while depending on corpus size...)"

    mallet train-topics \
        --num-threads $NUM_THREADS \
        --input "$OUTPUT_DIR/input.mallet" \
        --num-topics $NUM_TOPICS \
        --output-topic-keys "$OUTPUT_DIR/keys.txt" \
        --output-model "$OUTPUT_DIR/model.mallet" \
        --topic-word-weights-file "$OUTPUT_DIR/topic_word_weights.txt" \
        --word-topic-counts-file "$OUTPUT_DIR/word_topic_counts.txt" \
        --output-doc-topics "$OUTPUT_DIR/topics.txt" \
        --inferencer-filename "$OUTPUT_DIR/inferencer.mallet" \
        --optimize-interval $OPTIMIZE_INTERVAL \
        --diagnostics-file "$OUTPUT_DIR/diagnostics.xml" \
        --random-seed $RANDOM_SEED

    echo "  ✓ Topic model trained successfully"
    echo ""

    # Success message
    echo "================================================================================"
    echo "SUCCESS! Topic modeling complete."
    echo "================================================================================"
    echo ""
    echo "Output files in: $OUTPUT_DIR"
    echo ""
    echo "Key output files:"
    echo "  - keys.txt                   Topic keywords (human-readable)"
    echo "  - topics.txt                 Document-topic distributions"
    echo "  - topic_word_weights.txt     Word-topic distributions"
    echo "  - inferencer.mallet          For inference on new documents"
    echo "  - diagnostics.xml            Training diagnostics"
    echo ""
}

# Entry point
main
