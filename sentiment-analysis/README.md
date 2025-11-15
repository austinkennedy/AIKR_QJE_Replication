# Historical Text Sentiment Analysis Pipeline

## Project Structure

```
sentiment-analysis/
├── README.md                           # This file
├── sentiment_scorer.py                 # Main automated pipeline script
└── dictionaries/                       # Sentiment dictionaries
    ├── Sentiment Scores Other.csv      # Main progress dictionary (4 metrics)
    ├── Progress Scores Main.csv        # 1708 dictionary version (2 metrics)
    ├── ChatGPT Progress Dictionary.csv # AI-generated progress words
    ├── Industry and Optimism Dictionary.csv  # Industrial + optimism (2 metrics)
    ├── 1643 Dictionary.csv             # Weighted industrial dict (1643 cut)
    └── Appleby Dictionary.csv          # Full weighted industrial dict
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
python -c "import nltk"
```

## How It Works

The script runs a fully automated 6-step pipeline:

### Step 1: Configuration
Edit the configuration paths at the top of `sentiment_scorer.py` (lines 410-420):

```python
RAW_TEXT_DIR = r'f:/path/to/your/raw/text/files'
WORD_DIST_OUTPUT = r'f:/path/to/word/distributions'
OUTPUT_DIR = r'./output'
```

- `RAW_TEXT_DIR`: Directory containing your cleaned text files (.txt)
- `WORD_DIST_OUTPUT`: Directory to save/cache word distributions
- `OUTPUT_DIR`: Directory for final sentiment score CSV files

### Step 2: Generate Word Distributions 
- Reads each raw text file from `RAW_TEXT_DIR`
- Counts word frequencies (excludes words appearing only once)
- Saves CSV with columns: `word`, `count`, `pct`, `total_words`
- Skips files that already have distributions

### Step 3: Load Dictionaries 
- Loads 6 dictionary files containing 11 total metrics
- Applies Porter stemming where needed
- Separates into simple (9 metrics) and weighted (2 metrics) dictionaries

### Step 4: Create Volume Index 
- Indexes all word distribution files
- Converts filenames to HathiTrust IDs for tracking

### Step 5: Score All Volumes 
- Loads each volume CSV once
- Scores against all 11 metrics in memory
- Volume-first iteration: 264K disk reads 
- Processes ~50 volumes/second on typical hardware

### Step 6: Save Results 
- Splits results into 6 separate CSV files for analysis
- All files share the same index (volume filenames)
- Ready for downstream statistical analysis

## Usage

### Running the Pipeline

1. **Configure paths** in `sentiment_scorer.py`:

Open the script and edit the configuration section (around lines 410-420):

```python
# Configuration
RAW_TEXT_DIR = r'f:/path/to/your/raw/text/files'
WORD_DIST_OUTPUT = r'f:/path/to/word/distributions'
OUTPUT_DIR = r'./output'
```

2. **Run the script**:

```bash
cd sentiment-analysis
python sentiment_scorer.py
```

3. **Monitor progress**:

The script will display progress for each step:
- Step 1: Finding raw text files
- Step 2: Generating word distributions (with progress bar)
- Step 3: Loading dictionaries
- Step 4: Creating volume index
- Step 5: Scoring all volumes (with progress bar showing vol/sec)
- Step 6: Saving results

4. **Check results**:

Find 6 CSV files in `OUTPUT_DIR`:
- `Sentiment_scores_other.csv` - Progress, Optimism, Pessimism, Regression
- `progress_scores_main.csv` - Main and Secondary progress scores
- `Sentiment_ChatGPT.csv` - ChatGPT-generated progress scores
- `Optimism_abbr_industry_1708.csv` - Industrial and optimism scores
- `Industrialization_1643.csv` - Weighted industrial scores (1643 cutoff)
- `Industrialization_appleby.csv` - Full weighted industrial scores



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

For dictionaries with weights (Appleby Dictionary, 1643 Dictionary):

```
score = (sum of count × weight) / total_words
```

**Steps**:
1. Load dictionary with words and weights
2. Join with volume word distribution
3. Multiply word count by weight for each match
4. Sum all weighted counts
5. Divide by total words in the volume


### Quick Reference

| Dictionary File | Type | Metrics | Description |
|----------------|------|---------|-------------|
| Sentiment Scores Other.csv | Simple | 4 | Progress, Optimism, Pessimism, Regression |
| Progress Scores Main.csv | Simple | 2 | Main (post-1643), Secondary (1708 dict) |
| ChatGPT Progress Dictionary.csv | Simple | 1 | AI-generated progress dictionary |
| Industry and Optimism Dictionary.csv | Simple | 2 | Industrial prior + Optimism double meaning |
| 1643 Dictionary.csv | Weighted | 1 | 160 industrial terms with weights (1643 cut) |
| Appleby Dictionary.csv | Weighted | 1 | 207 industrial terms (full dictionary) |

## Word Distribution Methodology

1. **Read raw text** - UTF-8 encoded cleaned text files
2. **Split into words** - Whitespace-separated tokenization
3. **Count occurrences** - Group by word and sum counts
4. **Filter** - Keep only words appearing more than once
5. **Calculate percentages** - `pct = count / sum(counts)`
6. **Add metadata** - Include total_words column
7. **Save to CSV** - Word as index, count, pct, total_words columns


### Word Distribution Format

Each word distribution CSV contains:
- `word` (index): The word
- `count`: Number of occurrences 
- `pct`: Percentage of total word count
- `total_words`: Total word count for the volume

## Output Files

The script generates 6 separate CSV files optimized for downstream analysis:

| Output File | Source Dictionary | Columns | Description |
|------------|------------------|---------|-------------|
| Sentiment_scores_other.csv | Sentiment Scores Other.csv | Progress, Optimism, Pessimism, Regression | Core sentiment metrics |
| progress_scores_main.csv | Progress Scores Main.csv | Main, Secondary | Main and 1708 progress scores |
| Sentiment_ChatGPT.csv | ChatGPT Progress Dictionary.csv | ChatGPT_Progress | AI-generated progress metric |
| Optimism_abbr_industry_1708.csv | Industry and Optimism Dictionary.csv | Industrialization_Prior, Optimism_Double_Meaning | Industrial and optimism metrics |
| Industrialization_1643.csv | 1643 Dictionary.csv | Dict_1643 | Weighted industrial scores (1643 cutoff) |
| Industrialization_appleby.csv | Appleby Dictionary.csv | Appleby | Full weighted industrial scores |

All files share the same index (volume filenames) for easy merging.
