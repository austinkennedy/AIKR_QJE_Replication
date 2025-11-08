# AIKR
## Overview
The code in this replication package runs the analysis for the paper "Enlightenment Ideals and Beliefs in Progress in the Run-up to the Industrial Revolution: A Textual Analysis" by Ali Almelhem, Murat Iyigun, Austin Kennedy, and Jared Rubin. It produces 53 figures and 3 tables. The code in this replication package is split into four parts:

1. **Get Unique Volumes** - Volume deduplication from HTRC workset
2. **Preprocessing** - Text preprocessing and cleaning
3. **LDA** - MALLET topic modeling
4. **Final Analysis** - Statistical analysis, regressions, and visualization

The data for replication is available here: [Data For Replication](https://www.dropbox.com/scl/fo/2qg8lv11j41ytjp2ru3k7/AHfF5xuQUVdtFNYKjcwMBa0?rlkey=py6mt8kztk72g8ity4hqlpbqc&st=9d8zn45r&dl=0)

Detailed documentation on each part can be found within their respective directories.

---

## Complete Replication Pipeline

This repository provides the **complete four-stage pipeline** from initial HTRC workset to final figures and tables.

###  Pipeline Overview

```
HTRC Workset → [Get Unique Volumes] → Volume List → [Download] →
.json.bz2 Files → [Preprocessing] → Clean Text → [LDA] → Topics →
[Final Analysis] → Figures & Tables
```

### Four-Stage Process

**Stage 1: Get Unique Volumes** (Volume deduplication)
```bash
cd "Get Unique Volumes/"
jupyter notebook "Get Unique Volumes for Rsync.ipynb"
```

Deduplicates the initial HTRC workset (383K → 265K volumes), selecting complete serial sets. Outputs a list of unique volume IDs for download.

See [`Get Unique Volumes/README.md`](Get%20Unique%20Volumes/README.md) for detailed documentation.

**Stage 2: Text Preprocessing** (Clean HTRC data)
```bash
cd Preprocessing/
python preprocess_htrc.py
```

Converts HTRC Extracted Features files (`.json.bz2`) to cleaned text files (`.txt`). Applies 9-step pipeline including lemmatization, stopword removal, and modernization.

See [`Preprocessing/README.md`](Preprocessing/README.md) for detailed preprocessing documentation.

**Stage 3: Topic Modeling** (Train MALLET model)
```bash
cd LDA/
./mallet_LDA.sh --input-dir ./data --output-dir ./results
```

Trains MALLET LDA topic model (60 topics) on cleaned text. Produces document-topic distributions for analysis.

See [`LDA/README.md`](LDA/README.md) for detailed MALLET documentation.

**Stage 4: Final Analysis** (Statistical analysis and visualization)
```bash
cd final_analysis/
python main.py
```

Takes LDA output, performs data processing, runs regression analysis, and generates all 53 figures and 3 tables for the paper.

See [`final_analysis/README.md`](final_analysis/README.md) for detailed analysis documentation.


### Stage 1: Get Unique Volumes
- `Get Unique Volumes/` - Volume deduplication pipeline
  - `Get Unique Volumes for Rsync.ipynb` - Production deduplication notebook
  - `Get Unique Volumes.ipynb` - Original Jupyter notebook (reference)
  - See [`Get Unique Volumes/README.md`](Get%20Unique%20Volumes/README.md) for details

### Stage 2: Text Preprocessing
- `Preprocessing/` - HTRC text preprocessing pipeline
  - `preprocess_htrc.py` - Main preprocessing script (validated, production-ready)
  - `reference_data/` - Required dictionary files (corrections, modern/archaic mappings, etc.)
  - `test/` - Comprehensive test suite
  - See [`Preprocessing/README.md`](Preprocessing/README.md) for details

### Stage 3: MALLET Topic Modeling
- `LDA/` - MALLET topic modeling
  - `mallet_LDA.sh` - Main topic modeling script
  - `mallet_inference.sh` - Apply trained model to new documents
  - See [`LDA/README.md`](LDA/README.md) for details

### Stage 4: Final Analysis
- `final_analysis/` - Statistical analysis and visualization
  - `main.py` - Main analysis pipeline
  - `src/` - Analysis modules (categories, figures, regressions, etc.)
  - `Rscripts/` - R scripts for regression tables and additional figures
  - `configs/` - Configuration files
  - See [`final_analysis/README.md`](final_analysis/README.md) for details


## Description of Programs/Code

### Stage 1: Get Unique Volumes

The volume deduplication pipeline is implemented in Jupyter notebooks that process HTRC workset metadata to identify and select unique volumes while handling serial publications intelligently.

**Requirements:**
- Python 3.8+
- pandas
- tqdm (for progress bars)
- Jupyter Notebook

**Quick Start:**

```bash
cd "Get Unique Volumes/"
jupyter notebook "Get Unique Volumes for Rsync.ipynb"
```

**Output:**
- `deduplicated_volume_list.txt` - List of 264,825 unique volume IDs ready for download

For detailed documentation, see `Get Unique Volumes/README.md`.

### Stage 2: Preprocessing

The preprocessing pipeline converts HTRC Extracted Features files (`.json.bz2`) to cleaned text files ready for topic modeling.

**9-Step Processing Pipeline:**
1. Load HTRC Extracted Features (page-level word frequencies)
2. Spell correction using custom dictionary (1,255 corrections)
3. Modern/archaic word normalization (3,721 mappings)
4. Part-of-speech filtering (20 POS tags retained)
5. Minimum word length filter (≥2 characters)
6. Minimum word frequency filter (≥2 occurrences per volume)
7. Stopword removal
8. Lemmatization to base forms
9. Output sorted by word frequency


**Requirements:**
- Python 3.8+
- pandas
- numpy
- nltk (with WordNet corpus)
- htrc-feature-reader
- tqdm

**Quick Start:**

```bash
cd Preprocessing/

# Copy and edit configuration
cp config.template.sh config.sh
vim config.sh  # Set INPUT_DIR, OUTPUT_DIR

# Run preprocessing
python preprocess_htrc.py --input INPUT_DIR --output OUTPUT_DIR
```


**Output:**
- One `.txt` file per input volume
- Each line: `word frequency`
- Words sorted by frequency (descending)
- Empty files (0 bytes) for volumes with no clean text after filtering

For detailed documentation, see `Preprocessing/README.md`.

### Stage 3: LDA

The code stored in the `lda` directory implements MALLET (MAchine Learning for LanguagE Toolkit) topic modeling to extract 60 topics from the historical corpus. This produces the LDA output files that are used as input for the Final Analysis pipeline.

The topic modeling framework uses fixed model parameters for exact reproducibility:
- **Number of topics:** 60 (hardcoded)
- **Random seed:** 1 (hardcoded)
- **Optimization interval:** 500 (hardcoded)

Infrastructure parameters (paths, compute resources) are fully configurable via `lda/config.sh` file.

**Requirements:**
- MALLET installed and in PATH
- Java 1.8 or higher
- Bash 4.0+
- SLURM (optional, for HPC environments)

**Quick Start:**

1. Navigate to the LDA directory:
   ```bash
   cd lda
   ```

2. Create configuration file:
   ```bash
   cp config.template.sh config.sh
   vim config.sh  # Edit INPUT_DIR, OUTPUT_DIR, STOPLIST_FILE
   ```

3. Run topic modeling:
   ```bash
   # Local execution
   ./final_mallet_2025.sh

   # Or submit to SLURM
   sbatch final_mallet_2025.sh
   ```

**Output Files:**

The LDA pipeline produces several output files in the specified output directory:
- `topics.txt` - Document-topic distributions (main output for Final Analysis)
- `keys.txt` - Topic keywords
- `model.mallet` - Trained topic model
- `inferencer.mallet` - For applying model to new documents
- `diagnostics.xml` - Training diagnostics

**Documentation:**

For detailed documentation, see:
- `lda/README.md` - Complete documentation with all options
- `lda/QUICKSTART.md` - 30-second quick reference
- `lda/DEPLOY.md` - Step-by-step HPC deployment guide
- `lda/test/README_TESTING.md` - Testing framework documentation


### Stage 4: Final Analysis

The code stored in the `final_analysis` directory takes the output of the LDA model, performs data processing and transformation, runs the algorithm to get distinct "categories" of topics (e.g. Science, Religion, and Political Economy), runs regressions, and produces the final tables and figures.

This code was developed and run on Python 3.11.1. A `requirements.txt` file is provided listing dependencies and package versions. You can install it by navigating to the `final_analysis` directory in the command line and running:

```
pip install -r requirements.txt
```

**Quick Start**

To run the final analysis, first download the `final_analysis_input` folder from the data repository included above. Place this folder under `final_analysis/data/` (or alternatively, change the `input_path` parameter in the config files under `final_analysis/configs`).

The entirety of the final analysis can then be run by running the `final_analysis/main.py` script in Python:

**For Windows Users**
```
python final_analysis/main.py
```

**For MacOS/Linux Users**
```
python3 final_analysis/main.py
```

Note that all necessary input data (i.e. the output of the LDA model and additional files) must be present in order for the code to properly run. The code will create all figures and tables and store them in the `final_analysis/data/paper_assets/` directory.

Configuration files can be found under the `final_analysis/configs` directory.

For more detailed documentation, see `final_analysis/README.md`.


---

## Troubleshooting

### Stage 1: Get Unique Volumes
- **Issue**: Out of memory errors
  - **Solution**: Process metadata in chunks (adjust `chunksize` parameter)

### Stage 2: Preprocessing
- **Issue**: Script runs slowly
  - **Solution**: Check `--num-processes` (default: auto-detect cores)

- **Issue**: Missing NLTK data
  - **Solution**: Run `python -m nltk.downloader wordnet`

- **Issue**: Empty output files
  - **Solution**: This is expected behavior for volumes with no clean text after filtering (683 files, 0.26% of corpus)

### Stage 3: LDA
- **Issue**: Java heap space errors
  - **Solution**: Increase memory in config: `MEMORY=16G`

- **Issue**: MALLET command not found
  - **Solution**: Add MALLET to PATH: `export PATH=/path/to/mallet/bin:$PATH`

### Stage 4: Final Analysis
- **Issue**: Missing input data
  - **Solution**: Download from data repository and place in `final_analysis/data/`

- **Issue**: R script errors
  - **Solution**: Install required R packages listed in `final_analysis/Rscripts/r_config.yaml`

For additional troubleshooting, see individual stage README files.

---

## Acknowledgments

This analysis uses MALLET (MAchine Learning for LanguagE Toolkit) developed by Andrew McCallum.

**Citation for MALLET:**
> McCallum, Andrew Kachites. "MALLET: A Machine Learning for Language Toolkit."
> http://mallet.cs.umass.edu. 2002.

The historical text corpus is provided by the HathiTrust Research Center (HTRC).

**Citation for HTRC:**
> HathiTrust Research Center. "HTRC Extracted Features Dataset."
> https://analytics.hathitrust.org/

---

## License for Code

This code is licensed under the MIT license. See the LICENSE file for details.
