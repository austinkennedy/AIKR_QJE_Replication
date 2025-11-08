# Historical Text Sentiment Analysis Tool

A unified Python toolkit for calculating multi-dimensional sentiment scores from historical book volumes using various sentiment dictionaries. This tool consolidates multiple Jupyter notebook workflows into a single, maintainable script.

## Overview

This project analyzes historical book volumes from the HathiTrust Digital Library using multiple sentiment dictionaries to measure:
- **Progress** - Words indicating advancement and development
- **Optimism/Pessimism** - Positive and negative sentiment
- **Regression** - Words indicating decline or backward movement
- **Industrialization** - Terms related to industrial development
- **ChatGPT-derived Progress** - Modern AI-generated progress indicators

The tool supports both **simple (unweighted)** and **weighted** scoring methodologies, with results validated against original research outputs to ensure perfect accuracy.

## Features

- **Unified Sentiment Scoring**: Single script handles all dictionary types and methodologies
- **Word Distribution Generation**: Convert raw text to word frequency distributions
- **Multiple Dictionaries**: Support for 7 different sentiment dictionaries
- **Two Scoring Methods**:
  - Simple: Sum of percentage values for matching words
  - Weighted: Count × weight / total_words formula
- **Porter Stemming**: Automatic word normalization where needed
- **Batch Processing**: Efficiently process thousands of volumes

## Project Structure

```
sentiment-analysis/
├── README.md                           # This file
├── sentiment_scorer.py                 # Main unified scoring script
├── DICTIONARY_TO_OUTPUT_MAPPING.md     # Dictionary to output column mapping
└── dictionaries/                       # Sentiment dictionaries
    ├── README.md                       # Dictionary documentation
    ├── Updated Progress List.csv       # Main progress dictionary (4 metrics)
    ├── Updated Progress List May 2023.csv  # 1708 dictionary version
    ├── ChatGPT Progress Dictionary.csv # AI-generated progress words
    ├── Industrialization Dictionary (June 23).csv  # Simple industrial dict
    ├── Industrialization Dictionary (May 24).csv   # Weighted industrial dict (1643 cut)
    ├── Industry and Optimism Dictionary (May 2025).csv  # Industrial + optimism
    └── APPLEBY'S TOC (3-vote Threshold).csv  # Full weighted industrial dict
```

## Installation

### Requirements
- Python 3.7+
- pandas
- tqdm
- nltk

### Setup

```bash
# Install dependencies
pip install pandas tqdm nltk

# Download NLTK data (for Porter Stemmer)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### 1. Generate Word Distributions from Raw Text

Convert raw cleaned text files into word frequency distributions:

```python
from sentiment_scorer import generate_all_distributions

# Generate distributions for a list of files
source_dir = '/path/to/raw/text/files'
output_dir = './word_distributions'
filenames = ['book1.txt', 'book2.txt', 'book3.txt']

generate_all_distributions(source_dir, output_dir, filenames)
```

**Output format** (CSV with columns):
- `word` (index): The word
- `count`: Number of occurrences (must be > 1)
- `pct`: Percentage of total word count
- `total_words`: Total word count for the volume

### 2. Load Sentiment Dictionaries

```python
from sentiment_scorer import load_dictionaries

# Load all dictionaries
simple_dicts, weighted_dicts = load_dictionaries()

# simple_dicts contains: Progress, Optimism, Pessimism, Regression,
#                        Main, Secondary, ChatGPT_Progress, etc.
# weighted_dicts contains: APPLEBY_3vote, Industrial_May24
```

### 3. Score Individual Volumes

```python
from sentiment_scorer import score_volume_simple, score_volume_weighted

# Simple scoring (unweighted)
progress_score = score_volume_simple(
    './word_distributions/book1.txt',
    simple_dicts['Progress']
)

# Weighted scoring
industrial_score = score_volume_weighted(
    './word_distributions/book1.txt',
    weighted_dicts['Industrial_May24']
)
```

### 4. Batch Score All Volumes

```python
from sentiment_scorer import score_all_volumes

# Score all volumes across all dictionaries
results_df = score_all_volumes(simple_dicts, weighted_dicts)

# Save results
results_df.to_csv('sentiment_scores.csv')
```

**Output DataFrame structure**:
- Index: Filename
- Columns: One column per dictionary/metric (Progress, Optimism, Pessimism, etc.)
- Values: Sentiment scores (float)

## Scoring Methodologies

### Simple (Unweighted) Scoring

For dictionaries without weights (Progress, Optimism, Pessimism, Regression, etc.):

```
score = sum of pct values for all matching words
```

**Steps**:
1. Load dictionary words (apply Porter stemming if needed)
2. Join dictionary with volume word distribution on word index
3. Sum the `pct` column for matching words

### Weighted Scoring

For dictionaries with weights (APPLEBY's TOC, Industrial May24):

```
score = (sum of count × weight) / total_words
```

**Steps**:
1. Load dictionary with words and weights
2. Join with volume word distribution
3. Multiply word count by weight for each match
4. Sum all weighted counts
5. Divide by total words in the volume

## Dictionaries

See [dictionaries/README.md](dictionaries/README.md) for detailed descriptions of each dictionary.

### Quick Reference

| Dictionary File | Type | Metrics | Description |
|----------------|------|---------|-------------|
| Updated Progress List.csv | Simple | 4 | Progress, Optimism, Pessimism, Regression |
| Updated Progress List May 2023.csv | Simple | 2 | Main (post-1643), Secondary (1708 dict) |
| ChatGPT Progress Dictionary.csv | Simple | 1 | AI-generated progress indicators |
| Industrialization Dictionary (June 23).csv | Simple | 1 | 160 industrial terms (not used in main analysis) |
| Industrialization Dictionary (May 24).csv | Weighted | 1 | 160 industrial terms with weights (1643 cut) |
| Industry and Optimism Dictionary (May 2025).csv | Simple | 2 | Industrial prior + Optimism double meaning |
| APPLEBY'S TOC (3-vote Threshold).csv | Weighted | 1 | 207 industrial terms (full dictionary) |

## Word Distribution Methodology

The word distribution generation follows the exact methodology from original Jupyter notebooks:

1. **Read raw text** - UTF-8 encoded cleaned text files
2. **Split into words** - Whitespace-separated tokenization
3. **Count occurrences** - Group by word and sum counts
4. **Filter** - Keep only words appearing more than once
5. **Calculate percentages** - `pct = count / sum(counts)`
6. **Add metadata** - Include total_words column
7. **Save to CSV** - Word as index, count, pct, total_words columns

**Important**: Words appearing only once are excluded from distributions, matching the original notebook behavior.

## Validation


See [DICTIONARY_TO_OUTPUT_MAPPING.md](DICTIONARY_TO_OUTPUT_MAPPING.md) for mappings between dictionary files and original output columns.

## Output Column Naming

Results are saved with descriptive column names:

| Column Name | Dictionary | Metric |
|------------|-----------|--------|
| Progress | Updated Progress List | Progress sentiment |
| Optimism | Updated Progress List | Optimism sentiment |
| Pessimism | Updated Progress List | Pessimism sentiment |
| Regression | Updated Progress List | Regression sentiment |
| Main | Updated Progress List May 2023 | Main progress (post-1643) |
| Secondary | Updated Progress List May 2023 | Secondary progress (1708) |
| ChatGPT_Progress | ChatGPT Progress Dictionary | ChatGPT progress |
| Industrial_June23 | Industrialization Dictionary (June 23) | Simple industrial |
| Industrial_May24 | Industrialization Dictionary (May 24) | Weighted industrial (1643) |
| Industrialization_Prior | Industry and Optimism (May 2025) | Industrial prior |
| Optimism_Double_Meaning | Industry and Optimism (May 2025) | Optimism double meaning |
| APPLEBY_3vote | APPLEBY'S TOC | Full industrial dictionary |

## Technical Details


### Performance

- **Word distribution generation**: ~10-20 volumes/second (varies by file size)
- **Scoring**: ~100-130 volumes/second for simple dictionaries, ~100-120 for weighted
- **Memory efficient**: Processes files individually, suitable for large datasets

## Example Workflow

Complete workflow for processing new volumes:

```python
import pandas as pd
from sentiment_scorer import (
    generate_all_distributions,
    load_dictionaries,
    score_all_volumes
)

# Step 1: Generate word distributions
source_dir = '/path/to/cleaned/text'
output_dir = './word_distributions'

# Get list of all text files
import os
filenames = [f for f in os.listdir(source_dir) if f.endswith('.txt')]

# Generate distributions
generate_all_distributions(source_dir, output_dir, filenames)

# Step 2: Load dictionaries
simple_dicts, weighted_dicts = load_dictionaries()

# Step 3: Score all volumes
results = score_all_volumes(simple_dicts, weighted_dicts)

# Step 4: Save results
results.to_csv('sentiment_analysis_results.csv')

print(f"Processed {len(results)} volumes across {len(results.columns)} metrics")
```


