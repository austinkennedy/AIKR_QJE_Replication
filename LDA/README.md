# MALLET Topic Modeling - LDA Pipeline

Topic modeling framework using MALLET (MAchine Learning for LanguagE Toolkit) for text analysis research.


### Core Scripts
- `mallet_LDA.sh` - Main topic modeling script
- `mallet_inference.sh` - Apply trained model to new documents

### Configuration
- `config.template.sh` - Configuration template (copy to config.sh)
- `default_stoplist.txt` - Default stopword list (placeholder)

### Documentation
- `README.md` - This documentation
- `DEPLOY.md` - Step-by-step HPC deployment guide
- `package.sh` - Create deployment package

---

## Requirements

### Software
- **MALLET** installed and in PATH
- **Java** 1.8 or higher
- **Bash** 4.0+
- **SLURM** (optional, for HPC environments)

### Verify Installation
```bash
# Check MALLET
mallet --help

# Check Java
java -version

# Check SLURM (if using HPC)
sbatch --version
```

---


## Quick Start

### 1. Create Configuration File

```bash
cp config.template.sh config.sh
vim config.sh  # Edit with your paths
```

Required settings in `config.sh`:
```bash
INPUT_DIR="/path/to/your/documents"
OUTPUT_DIR="/path/to/your/results"
```

Optional settings:
```bash
STOPLIST_FILE="/path/to/stoplist.txt"  # Leave empty for default
NUM_THREADS="48"                        # Leave empty for auto-detect
```

### 2. Make Scripts Executable

```bash
chmod +x mallet_LDA.sh mallet_inference.sh
```

### 3. Run Topic Modeling

**Using config file (recommended):**
```bash
./mallet_LDA.sh
```

**Or using command-line arguments:**
```bash
./mallet_LDA.sh \
    --input-dir ./my_documents \
    --output-dir ./results
```

**For HPC/SLURM:**
```bash
sbatch mallet_LDA.sh
```

### 4. View Results

Results are saved in your output directory:
- `topics.txt` - Document-topic distributions (main output)
- `keys.txt` - Topic keywords (human-readable)
- `model.mallet` - Trained topic model
- `inferencer.mallet` - For applying to new documents
- `diagnostics.xml` - Training diagnostics

---

## Configuration

### Configuration Precedence

Settings are applied in this order (later overrides earlier):
1. **Script defaults** (lowest priority)
2. **Config file** (`config.sh`)
3. **Command-line arguments** (highest priority)

### Configuration File

```bash
# 1. Copy template
cp config.template.sh config.sh

# 2. Edit with your settings
vim config.sh

# 3. Run (config file used automatically)
./mallet_LDA.sh
```

### Key Settings

**Main Script (`mallet_LDA.sh`):**
```bash
INPUT_DIR="/path/to/input/documents"     # Required: Text files directory
OUTPUT_DIR="/path/to/output/results"     # Required: Results directory
STOPLIST_FILE="/path/to/stoplist.txt"    # Optional: Stopwords file
NUM_THREADS="48"                         # Optional: CPU threads (auto-detect if empty)
```

**Inference Script (`mallet_inference.sh`):**
```bash
INFERENCER_FILE="/path/to/inferencer.mallet"  # From training run
INFERENCE_INPUT="/path/to/new/documents"      # New documents
INFERENCE_OUTPUT="/path/to/output.txt"        # Output file
INFERENCE_RANDOM_SEED="1"                     # Default: 1
```

---

## Usage

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-dir <path>` | Directory containing text files | Required |
| `--output-dir <path>` | Directory for output files | Required |
| `--num-threads <n>` | Number of threads | Auto-detect |
| `--dry-run` | Preview commands without executing | Off |
| `--help, -h` | Show help message | - |

---

## Output Files

After successful completion, your output directory will contain:

| File | Description | Size | Usage |
|------|-------------|------|-------|
| `topics.txt` | Document-topic distributions | Large | Main analysis input |
| `keys.txt` | Topic keywords | Small | Human interpretation |
| `model.mallet` | Trained topic model | Large | Model persistence |
| `inferencer.mallet` | Inference model | Large | Apply to new docs |
| `diagnostics.xml` | Training diagnostics | Small | Model evaluation |

### File Formats

**topics.txt:**
```
doc_id topic_0_weight topic_1_weight ... topic_59_weight
```

**keys.txt:**
```
topic_id alpha words...
```

---

## Inference on New Documents

Apply your trained model to new documents:

### 1. Configure Inference

Edit `config.sh`:
```bash
INFERENCER_FILE="/path/to/results/inferencer.mallet"
INFERENCE_INPUT="/path/to/new/documents"
INFERENCE_OUTPUT="/path/to/new_topics.txt"
```

### 2. Run Inference

```bash
./mallet_inference.sh
```

Or with command-line arguments:
```bash
./mallet_inference.sh \
    --inferencer ./results/inferencer.mallet \
    --input ./new_documents \
    --output ./new_topics.txt
```

### Example Workflow

```bash
# Step 1: Train model
./mallet_LDA.sh \
    --input-dir ./training_data \
    --output-dir ./model_output

# Step 2: Apply to new documents
./mallet_inference.sh \
    --inferencer ./model_output/inferencer.mallet \
    --input ./new_documents \
    --output ./new_topics.txt
```

---

## HPC/SLURM Usage

### 1. Edit Configuration

```bash
cp config.template.sh config.sh
vim config.sh
```

Example for HPC:
```bash
INPUT_DIR="/pl/active/lab/data"
OUTPUT_DIR="/pl/active/lab/results_$(date +%Y%m%d)"
STOPLIST_FILE="/pl/active/lab/stoplist.txt"
NUM_THREADS="48"
```

### 2. Customize SLURM Headers

Edit `mallet_LDA.sh` (lines 12-20):
```bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=500000                # Memory in MB
#SBATCH --time=40:00:00             # Max runtime
#SBATCH --ntasks=48                 # CPU cores
```

### 3. Submit Job

```bash
sbatch mallet_LDA.sh
```

### 4. Monitor Job

```bash
# Check status
squeue -u $USER

# Watch output
tail -f mallet_run_*.out

# Check detailed status
scontrol show job JOBID
```

---

## Troubleshooting

### "Input directory is empty"

**Problem:** No text files in input directory.

**Solution:**
- Verify path: `ls -la /path/to/documents`
- Check for `.txt` files: `ls /path/to/documents/*.txt | wc -l`
- Verify files have content: `head ./documents/doc001.txt`

### "Permission denied"

**Problem:** Script is not executable.

**Solution:**
```bash
chmod +x mallet_LDA.sh mallet_inference.sh
```

### Out of Memory Errors

**Problem:** Insufficient memory for large corpus.

**Solution for HPC:**
```bash
# Edit SLURM header in mallet_LDA.sh
#SBATCH --mem=500000  # Increase this value (in MB)
```

**Solution for local machine:**
```bash
# Set MALLET memory before running
export MALLET_MEMORY=8g
./mallet_LDA.sh --input-dir ./data --output-dir ./results
```

### Module Load Errors (HPC)

**Problem:** Java module not available on your cluster.

**Solution:**
```bash
# Check available modules
module avail java
module avail jdk

# Edit script (line 57) to use correct module name
MODULE_JAVA="java/11"  # Or whatever is available

# Or load manually before running
module load java/11
./mallet_LDA.sh --input-dir ./data --output-dir ./results
```

### Very Slow Training

**Problem:** Topic training is taking too long.

**Possible causes and solutions:**
1. **Large corpus:** This is expected. Check progress in SLURM output file.
2. **Wrong thread count:** Verify `NUM_THREADS` matches available cores
3. **Disk I/O bottleneck:** Ensure input/output on fast storage (not NFS if possible)
4. **Memory swapping:** Increase `--mem` in SLURM header

---

## Advanced Topics

### Testing on Small Dataset

Use `--dry-run` to preview commands without executing:

```bash
./mallet_LDA.sh \
    --input-dir ./data \
    --output-dir ./results \
    --dry-run
```

This shows exactly what will be executed without actually running the analysis.


## File Structure

```
LDA/
├── mallet_LDA.sh              Main topic modeling script
├── mallet_inference.sh        Apply model to new documents
├── config.template.sh         Configuration template (commit this)
├── config.sh                  Your configuration (gitignored)
├── default_stoplist.txt       Default stopword list
├── package.sh                 Create deployment package
├── DEPLOY.md                  HPC deployment guide
└── README.md                  This documentation
```


