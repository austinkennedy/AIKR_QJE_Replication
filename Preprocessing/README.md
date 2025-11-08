# HTRC Text Preprocessing for Topic Modeling

This directory contains the preprocessing pipeline for HTRC Extracted Features files, producing cleaned text suitable for MALLET topic modeling.

---

## Purpose

This preprocessing script is part of the replication package that transforms raw HTRC Extracted Features data into clean text files ready for topic modeling.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Processing Pipeline](#processing-pipeline)
5. [Reference Data Files](#reference-data-files)
6. [Output Format](#output-format)
7. [Integration with MALLET](#integration-with-mallet)
8. [Command-Line Reference](#command-line-reference)
9. [Troubleshooting](#troubleshooting)
10. [Testing](#testing)

---

## Requirements

### Python Version
- Python 3.7 or higher

### Python Libraries

Install required packages:
```bash
pip install pandas numpy tqdm htrc-features nltk pycountry
```

### NLTK Data (Required)

Download required NLTK datasets:
```bash
python -m nltk.downloader words names stopwords wordnet
```

This downloads:
- `words` - English dictionary
- `names` - People's names
- `stopwords` - Common stopwords
- `wordnet` - For lemmatization

### Input Data

- HTRC Extracted Features files (`.json.bz2` format)
- Available from [HathiTrust Research Center](https://analytics.hathitrust.org/)

---

## Quick Start

### 1. Configure

```bash
# Copy template
cp config.template.sh config.sh

# Edit with your paths
vim config.sh
```

Set these required fields in `config.sh`:
```bash
INPUT_DIR="/path/to/htrc_files"
OUTPUT_DIR="/path/to/cleaned_output"
```

### 2. Run

```bash
# Make sure you're in the Preprocessing directory
cd Preprocessing/

# Run preprocessing
python preprocess_htrc.py --config config.sh
```

### 3. Verify Output

```bash
# Check output files
ls -lh /path/to/cleaned_output/

# Each input .json.bz2 file produces one .txt file
```

---

## Configuration

### Configuration File (Recommended)

The recommended approach is to use a configuration file:

```bash
# 1. Copy template
cp config.template.sh config.sh

# 2. Edit config.sh with your paths
vim config.sh

# 3. Run with config
python preprocess_htrc.py --config config.sh
```

### Command-Line Arguments

Alternatively, specify paths via command line:

```bash
python preprocess_htrc.py \
    --input /path/to/htrc_files \
    --output /path/to/cleaned_output \
    --dict-corrections ./reference_data/Master_Corrections.csv \
    --dict-ma ./reference_data/MA_Dict_Final.csv \
    --world-cities ./reference_data/world_cities.csv
```

### Configuration Precedence

Settings are applied in this order (later overrides earlier):
1. Config file defaults
2. `config.sh` settings
3. Command-line arguments (highest priority)

---

## Processing Pipeline

The preprocessing pipeline consists of 8 steps applied to each HTRC volume:

### Step 1: Token Extraction

Extract tokens from HTRC Extracted Features format.

**Input:** `.json.bz2` files (HTRC format)
**Filter:** Retain only tokens with specific POS tags (20 tags)

**POS Tags Retained:**
- Nouns: `NE, NN, NNP, NNPS`
- Adjectives: `JJ, JJS, JJR`
- Verbs: `VB, VBP, VBZ, VBD, VBN, VBG`
- Adverbs: `RB, RBR, RBS`
- Other: `IN, DT, RP, CC`

### Step 2: Punctuation Cleaning

Remove all non-alphabetic characters.

**Transformations:**
- Remove: `,():-—;"!?•$%@""#<>+=/[]*^\'{}_■~\\|«»©&~£·`
- Delete all non-alpha characters

### Step 3: Character Corrections

Fix OCR errors and normalize Unicode.

**Greek Character Corrections:**
- `º` → `o`
- `ª` → `a`
- `ſ` → `s` (long s)
- `β` → `b`

**Ligature Fixes:**
- `ﬁ` → `fi`
- `ﬂ` → `fl`
- `ﬀ` → `ff`
- `ﬅ` → `ft`
- `ﬃ` → `ffi`
- `ﬄ` → `ffl`

**Unicode Normalization:**
- Apply NFKC normalization

### Step 4: Spelling Corrections

Apply dictionary-based spelling corrections.

**Dictionary:** `Master_Corrections.csv` (~19KB, curated corrections)

Common corrections:
- OCR errors: `teh` → `the`
- Historical spellings: `connexion` → `connection`
- Variants: `recieve` → `receive`

### Step 5: Minimum Thresholds

Filter tokens by length and frequency.

**Filters:**
- Minimum word length: **2 characters**
- Minimum frequency: **2 occurrences per volume**

### Step 6: Lemmatization/Stemming

Normalize word forms using POS-aware lemmatization.

**Process:**
1. Try POS-aware lemmatization (NLTK WordNetLemmatizer)
2. If unchanged, try lemmatization without POS
3. If still unchanged, try stemming (Snowball Stemmer)
4. Accept stemmed form if it appears in reference dictionary (cities, countries, names, English stopwords, modern words, continents, stems, days, months, Roman numerals)

### Step 7: Modern/Archaic Mapping

Standardize archaic language to modern equivalents.

**Dictionary:** `MA_Dict_Final.csv` (~73KB)

Examples:
- `whilst` → `while`
- `hath` → `has`
- `unto` → `to`
- `doth` → `does`

### Step 8: Stopword Filtering

Remove common function words and numeric markers using 2 categories:

| Category | Source | Count | Purpose |
|----------|--------|-------|---------|
| **English Stopwords** | NLTK stopwords | ~179 | Common words (the, a, is, etc.) |
| **Roman Numerals** | Generated (0-500) | 501 | Numeric markers (i, ii, iii, iv, etc.) |



### Step 9: Final Stemming

Apply final stemming pass for consistency.

**Result:** Cleaned, normalized, filtered tokens ready for topic modeling.

---

## Reference Data Files

Three dictionary files are required and included in `reference_data/`:

### 1. Master_Corrections.csv
- **Size:** ~19 KB
- **Purpose:** Spelling corrections for OCR errors
- **Format:** `lemma,correct_spelling`

### 2. MA_Dict_Final.csv
- **Size:** ~73 KB
- **Purpose:** Modern/Archaic word mappings
- **Format:** `orig,stand`

### 3. world_cities.csv
- **Size:** ~993 KB
- **Purpose:** Geographic stopword filtering
- **Format:** `name,country,subcountry,...`

See [`reference_data/README.md`](reference_data/README.md) for detailed documentation.

---

## Output Format


### File Content

Each output file contains space-separated tokens:
```
progress progress progress enlightenment enlightenment reason science philosophy nature virtue liberty freedom
```

**Word Repetition:** Words are repeated according to their frequency in the source volume.

**Format:** Plain text, UTF-8 encoding, one volume per file.

### Output Statistics

After processing, you'll see:
```
Processed 1234/1234 volumes successfully

SUCCESS! Preprocessing complete
Cleaned text files: /path/to/output
Total volumes: 1234
```

---

## Integration with MALLET

This preprocessing step is designed to produce text files ready for MALLET topic modeling.

### Complete Workflow

```bash
# Step 1: Preprocessing (this directory)
cd Preprocessing/
python preprocess_htrc.py --config config.sh

# Step 2: Topic Modeling (parent directory)
cd ..
./LDA/mallet_LDA.sh \
    --input-dir ./Preprocessing/output_cleaned \
    --output-dir ./results
```

### Data Flow

```
HTRC .json.bz2 → [preprocess_htrc.py] → Cleaned .txt → [MALLET] → Topic Model
```

---

## Command-Line Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Input directory with `.json.bz2` files |
| `--output` | Output directory for `.txt` files |
| `--dict-corrections` | Path to `Master_Corrections.csv` |
| `--dict-ma` | Path to `MA_Dict_Final.csv` |
| `--world-cities` | Path to `world_cities.csv` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Configuration file (shell format) | None |
| `--num-processes` | CPU processes | Auto-detect |
| `--error-log` | Error log file | stderr only |
| `--dry-run` | Preview without executing | Off |
| `--verbose, -v` | Verbose output | Off |
| `--help, -h` | Show help message | - |

### Examples

**Using config file:**
```bash
python preprocess_htrc.py --config config.sh
```

**Using command-line arguments:**
```bash
python preprocess_htrc.py \
    --input ./htrc_data \
    --output ./cleaned_text \
    --dict-corrections ./reference_data/Master_Corrections.csv \
    --dict-ma ./reference_data/MA_Dict_Final.csv \
    --world-cities ./reference_data/world_cities.csv
```

**Preview without running:**
```bash
python preprocess_htrc.py --config config.sh --dry-run
```

**Override output directory:**
```bash
python preprocess_htrc.py --config config.sh --output /different/path
```

**Specify CPU processes:**
```bash
python preprocess_htrc.py --config config.sh --num-processes 16
```

---

## Troubleshooting

### "No module named 'htrc_features'"

**Problem:** Required Python library not installed.

**Solution:**
```bash
pip install htrc-features
```

### "NLTK data not found"

**Problem:** Required NLTK datasets not downloaded.

**Solution:**
```bash
python -m nltk.downloader words names stopwords wordnet
```

### "Dictionary file not found"

**Problem:** Reference data files are missing.

**Solution:**
```bash
# Verify files exist
ls -l reference_data/

# Should see:
# Master_Corrections.csv
# MA_Dict_Final.csv
# world_cities.csv

# If missing, check git clone was complete
```

### "No .json.bz2 files found"

**Problem:** Input directory doesn't contain HTRC files.

**Solution:**
1. Verify input path: `ls /path/to/input`
2. Check for `.json.bz2` files: `find /path/to/input -name "*.json.bz2" | head`
3. Ensure you're pointing to the correct directory

### Out of Memory Errors

**Problem:** Insufficient memory for processing.

**Solution:**
```bash
# Reduce number of parallel processes
python preprocess_htrc.py --config config.sh --num-processes 4
```

### Very Slow Processing

**Problem:** Processing taking too long.

**Possible causes and solutions:**
1. **Large corpus:** This is expected. Monitor progress bar.
2. **Too few processes:** Increase: `--num-processes 16`
3. **Slow disk:** Use local SSD instead of network storage
4. **Limited memory:** Reduce processes to avoid swapping

### Permission Errors

**Problem:** Cannot write to output directory.

**Solution:**
```bash
# Check permissions
ls -ld /path/to/output

# Create directory with correct permissions
mkdir -p /path/to/output
chmod 755 /path/to/output
```

---

## Testing

### Running Tests

```bash
cd test/
./run_tests.sh
```

### Test Coverage

- Character cleaning (Greek, ligatures, Unicode)
- Roman numeral generation
- Processing parameter immutability
- Dictionary loading
- Configuration parsing

### Adding Sample Data

**For testing with real HTRC data:**
1. Place 2-3 small `.json.bz2` files in `test/sample_data/`
2. Run integration tests

---

## Advanced Topics

### Monitoring Progress

For large datasets, monitor progress:

```bash
# Watch process
ps aux | grep preprocess_htrc.py

# Monitor output directory
watch "ls -lh /path/to/output | tail -20"

# Count processed files
watch "ls /path/to/output/*.txt | wc -l"
```

### Debugging Individual Volumes

To process a single volume for debugging:

```python
from preprocess_htrc import *
from htrc_features import FeatureReader

# Load single volume
volume = FeatureReader(['path/to/volume.json.bz2']).first()

# Process
clean_df = process_volume_pipeline(volume)
print(clean_df.head(20))
```

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Verify all [Requirements](#requirements) are met
3. Try `--dry-run` to preview: `python preprocess_htrc.py --config config.sh --dry-run`
4. Check logs for specific errors
5. Review HTRC documentation: https://analytics.hathitrust.org/

---

## File Structure

```
Preprocessing/
├── preprocess_htrc.py          Main preprocessing script
├── config.template.sh          Configuration template
├── README.md                   This documentation
├── CHANGELOG.md                Version history and changes
├── reference_data/             Dictionary files
│   ├── README.md
│   ├── Master_Corrections.csv
│   ├── MA_Dict_Final.csv
│   └── world_cities.csv
└── test/                       Testing infrastructure
    ├── run_tests.sh
    ├── test_preprocessing.py
    └── sample_data/            Sample HTRC files
```

---


## References

- **HTRC Extracted Features:** https://analytics.hathitrust.org/
- **HTRC Feature Reader Documentation:** https://htrc.github.io/htrc-feature-reader/
- **NLTK Documentation:** https://www.nltk.org/
- **Topic Modeling with MALLET:** ../README.md

---

## Acknowledgments

This preprocessing pipeline was developed for analyzing historical texts from the HathiTrust Digital Library.

**Citation:**
> Almelhem, Ali, Murat Iyigun, Austin Kennedy, and Jared Rubin. "Enlightenment Ideals and Beliefs in Progress in the Run-up to the Industrial Revolution: A Textual Analysis." Quarterly Journal of Economics (forthcoming).
