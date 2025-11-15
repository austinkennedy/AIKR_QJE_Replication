# MALLET Topic Modeling - LDA Pipeline

Topic modeling framework using MALLET (MAchine Learning for LanguagE Toolkit) for text analysis research.


### Core Scripts
- `mallet_LDA.sh` - Main topic modeling script

### Configuration
- Edit `mallet_LDA.sh` directly to configure paths
- Uses `words_to_delete.txt` as the default stoplist

### Documentation
- `README.md` - This documentation

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

### 1. Configure Script

Edit `mallet_LDA.sh` and set your paths (lines 50-51):

```bash
INPUT_DIR="/path/to/your/documents"
OUTPUT_DIR="/path/to/your/results"
```

Optional settings (line 54):
```bash
NUM_THREADS="48"                        # Leave empty for auto-detect
```

### 2. Make Script Executable

```bash
chmod +x mallet_LDA.sh
```

### 3. Run Topic Modeling

```bash
./mallet_LDA.sh
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

Edit `mallet_LDA.sh` and modify the configuration section (lines 50-54):

```bash
# REQUIRED: Edit these paths for your environment
INPUT_DIR="/path/to/input/documents"     # Where your cleaned text files are
OUTPUT_DIR="/path/to/output/results"     # Where results will be saved

# OPTIONAL: Customize as needed
NUM_THREADS="48"                         # CPU threads (leave empty for auto-detect)
```


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

## HPC/SLURM Usage

### 1. Edit Script Configuration

Edit `mallet_LDA.sh` and set your paths (lines 50-51, 54):

```bash
INPUT_DIR="/pl/active/lab/data"
OUTPUT_DIR="/pl/active/lab/results_$(date +%Y%m%d)"
NUM_THREADS="48"
```

### 2. Customize SLURM Headers

Edit `mallet_LDA.sh` SLURM headers (lines 12-20):
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
chmod +x mallet_LDA.sh
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
./mallet_LDA.sh
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
./mallet_LDA.sh
```

### Very Slow Training

**Problem:** Topic training is taking too long.

**Possible causes and solutions:**
1. **Large corpus:** This is expected. Check progress in SLURM output file.
2. **Wrong thread count:** Verify `NUM_THREADS` matches available cores
3. **Disk I/O bottleneck:** Ensure input/output on fast storage (not NFS if possible)
4. **Memory swapping:** Increase `--mem` in SLURM header

---

## File Structure

```
LDA/
├── mallet_LDA.sh              Main topic modeling script
├── words_to_delete.txt        Default stoplist
└── README.md                  This documentation
```


