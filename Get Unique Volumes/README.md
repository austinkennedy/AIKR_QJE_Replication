# Get Unique Volumes - Stage 1: Volume Deduplication

This folder contains the first stage of the three-stage MALLET topic modeling pipeline. 

## Overview

HathiTrust volumes often have multiple digitized copies from different university libraries. This stage selects one copy of each work:
- **Unique volumes**: Keep the only available copy
- **Non-serial duplicates**: Select any copy (all are equivalent)
- **Serial publications**: Select the most complete set from a single university, then fill gaps with volumes from other universities

## Quick Start

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages: `pandas`, `tqdm`

Install dependencies:
```bash
pip install pandas tqdm jupyter
```

### Required Input Files

1. **htrc_workset_final.csv** (75 MB)
   - Initial workset provided by HTRC Librarian
   - Contains ~383K volumes printed in England between 1500-1900

2. **hathi_full_20241001.txt.gz** (1.1 GB)
   - HathiTrust metadata file
   - Use the most recent monthly release

Download from: [Dropbox](https://www.dropbox.com/scl/fo/2qg8lv11j41ytjp2ru3k7/AHfF5xuQUVdtFNYKjcwMBa0?rlkey=py6mt8kztk72g8ity4hqlpbqc&st=9d8zn45r&dl=0)

### Running the Notebook

1. Place input files in this folder
2. Open the notebook:
   ```bash
   jupyter notebook "Get Unique Volumes for Rsync.ipynb"
   ```
3. Run all cells (Runtime: ~2 minutes)

The notebook will generate:
- **deduplicated_volume_list.txt** - List of 264,825 unique volumes

## Output File Format

The output file `deduplicated_volume_list.txt` contains one volume ID (htid) per line:

```
volume
hvd.32044010273006
mdp.39015003646767
uc1.b3655527
...
```

## Next Steps: Downloading Volume Data with HTRC

### 1. Upload to HTRC Analytics

1. Go to: https://analytics.hathitrust.org/
2. Navigate to **Extracted Features Download Helper**
3. Upload `deduplicated_volume_list.txt`

### 2. Generate rsync Script

**IMPORTANT**: Select the correct database version:
- Choose **features-2020.03** as the data source
- This ensures the generated rsync script uses the correct server path:
  ```bash
  --files-from=- data.analytics.hathitrust.org::features-2020.03
  ```

The Download Helper will generate a bash script (e.g., `htrc_download.sh`)

### 3. Fix Dollar Sign Character Encoding

**IMPORTANT**: The rsync script requires special handling for volume IDs containing the `$` character.

After generating the script, you need to modify how these IDs are processed. Volume IDs like `uc1.$b620359` need to be escaped as `uc1/\$b620359/uc1.\$b620359.json.bz2`

Example of the issue:
```
# Volume ID in our list:
uc1.$b620359

# Correct path in rsync:
uc1/\$b620359/uc1.\$b620359.json.bz2

# Incorrect (without escaping):
uc1/$b620359/uc1.$b620359.json.bz2  # Will fail
```

Make sure the rsync script properly escapes all `$` characters as `\$` when constructing file paths. A simple 'find and replace' will solve this problem. 

### 4. Run rsync Script

Once the script is corrected, run it to download all `.json.bz2` files:

```bash
bash htrc_download.sh
```

This will download ~265K compressed JSON files containing extracted features for each volume.

## Pipeline Flow

```
HTRC Workset (383K volumes)
    ↓
[Stage 1: Get Unique Volumes] ← YOU ARE HERE
    ↓
deduplicated_volume_list.txt (265K volumes)
    ↓
[HTRC Download Helper] → rsync script
    ↓
.json.bz2 files (extracted features)
    ↓
[Stage 2: Preprocessing] → Clean text
    ↓
[Stage 3: MALLET LDA] → Topic model
```

## Deduplication Algorithm

### Step-by-Step Process

1. **Filter workset** by year (1500-1900) and language (English)
2. **Separate unique volumes** (no duplicates) 
3. **Load metadata** for duplicated volumes only
4. **Identify non-serial volumes** (empty description field) → select first copy
5. **Identify serial volumes** (non-empty description field)
6. **Standardize descriptions** (v.1, vol.1, V1 → v.1)
7. **Count volumes per source** for each serial
8. **Select best source** (university with most complete set)
9. **Fill gaps** with volumes from other universities
10. **Combine all groups** and export

### Serial Publication Handling

For multi-volume works (e.g., a 10-volume encyclopedia):
1. Count how many volumes each university has
2. Select the university with the most volumes
3. If no university has all volumes, fill gaps from other universities
4. This ensures the most complete set possible

## Files in This Folder

### Main Files
- **Get Unique Volumes for Rsync.ipynb** - Production notebook
- **deduplicated_volume_list.txt** - Output file (generated)

### Input Files (not in repo)
- **htrc_workset_final.csv** - Initial workset
- **hathi_full_20241001.txt.gz** - HathiTrust metadata

### Reference Files
- **EF_Rsync_deduplicated.sh** - Our rsync script


## Contact

For questions about the MALLET topic modeling pipeline, refer to the main repository README.

For HathiTrust-specific questions:
- HTRC Analytics: https://analytics.hathitrust.org/
- HathiTrust Research Center: https://www.hathitrust.org/htrc
