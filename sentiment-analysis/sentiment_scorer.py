"""
Unified Sentiment Scoring Script

This script calculates sentiment scores for all volumes using various dictionaries.
"""

import pandas as pd
from tqdm import tqdm
import os
from nltk.stem.porter import PorterStemmer


def load_dictionaries():
    """
    Load all sentiment dictionaries and organize them into two structures:
    1. simple_dicts: For unweighted scoring (sum of pct values)
    2. weighted_dicts: For weighted scoring (count × weight / total_words)

    Returns:
        tuple: (simple_dicts, weighted_dicts)
            - simple_dicts: dict of DataFrames with word index only
            - weighted_dicts: dict of DataFrames with word index and 'count' column
    """
    stemmer = PorterStemmer()
    dict_path = r'./dictionaries'

    simple_dicts = {}
    weighted_dicts = {}

    print("\nLoading dictionaries...")

    # 1. Sentiment Scores Other - 4 columns (progress, optimism, pessimism, regression)
    print("  Loading: Sentiment Scores Other")
    df_words = pd.read_csv(os.path.join(dict_path, 'Sentiment Scores Other.csv'), header=None)
    for col_idx, col_name in [(0, 'Progress'), (1, 'Optimism'), (2, 'Pessimism'), (3, 'Regression')]:
        df = pd.DataFrame()
        df['word'] = [stemmer.stem(x) for x in df_words[col_idx].dropna()]
        df = df.set_index('word')
        simple_dicts[col_name] = df
        print(f"    - {col_name}: {len(df)} words")

    # 2. Progress Scores Main - 2 columns (Main, Secondary)
    print("  Loading: Progress Scores Main")
    df_words = pd.read_csv(os.path.join(dict_path, 'Progress Scores Main.csv'), header=None)
    for col_idx, col_name in [(0, 'Main'), (1, 'Secondary')]:
        df = pd.DataFrame()
        df['word'] = [stemmer.stem(x) for x in df_words[col_idx].dropna()]  # Include all rows
        df = df.set_index('word')
        simple_dicts[col_name] = df
        print(f"    - {col_name}: {len(df)} words")

    # 3. ChatGPT Progress Dictionary - 1 column
    print("  Loading: ChatGPT Progress Dictionary")
    df_words = pd.read_csv(os.path.join(dict_path, 'ChatGPT Progress Dictionary.csv'))
    df = pd.DataFrame()
    df['word'] = [stemmer.stem(x) for x in df_words['ChatGPT_Progress'].dropna()]
    df = df.set_index('word')
    simple_dicts['ChatGPT_Progress'] = df
    print(f"    - ChatGPT_Progress: {len(df)} words")

    # 4. Industry and Optimism Dictionary - 2 columns
    print("  Loading: Industry and Optimism Dictionary")
    df_words = pd.read_csv(os.path.join(dict_path, 'Industry and Optimism Dictionary.csv'))
    for col, dict_name in [('Industrialization Prior', 'Industrialization_Prior'),
                           ('Optimism Double Meaning', 'Optimism_Double_Meaning')]:
        df = pd.DataFrame()
        df['word'] = [stemmer.stem(x) for x in df_words[col].dropna()]
        df = df.set_index('word')
        simple_dicts[dict_name] = df
        print(f"    - {dict_name}: {len(df)} words")

    # 5. Appleby Dictionary - WEIGHTED, already stemmed
    print("  Loading: Appleby Dictionary [WEIGHTED]")
    df = pd.read_csv(os.path.join(dict_path, 'Appleby Dictionary.csv'))
    df = df.set_index('word')
    weighted_dicts['Appleby'] = df
    print(f"    - Appleby: {len(df)} words (weighted)")

    # 6. 1643 Dictionary - WEIGHTED, already stemmed
    print("  Loading: 1643 Dictionary [WEIGHTED]")
    df = pd.read_csv(os.path.join(dict_path, '1643 Dictionary.csv'))
    df = df.set_index('word')
    weighted_dicts['Dict_1643'] = df
    print(f"    - Dict_1643: {len(df)} words (weighted)")

    print(f"\nLoaded {len(simple_dicts)} simple dictionaries and {len(weighted_dicts)} weighted dictionaries")

    return simple_dicts, weighted_dicts


def score_volume_simple(volume_path, dict_df):
    """
    Calculate simple (unweighted) sentiment score for a volume.

    Methodology: Sum of pct (percentage) values for matching words

    Args:
        volume_path: Path to volume word distribution CSV
        dict_df: Dictionary DataFrame with word index only

    Returns:
        float: Sentiment score (sum of word percentages)
    """
    # Load volume word distribution
    df_vol = pd.read_csv(volume_path).set_index('word')

    # Join dictionary with volume (left join keeps only dictionary words)
    df_joined = dict_df.join(df_vol, how='left').fillna(0)

    # Sum the percentages
    score = df_joined['pct'].sum()

    return score


def score_volume_weighted(volume_path, dict_df):
    """
    Calculate weighted sentiment score for a volume.

    Methodology: (Sum of count × weight) / total_words

    Args:
        volume_path: Path to volume word distribution CSV
        dict_df: Dictionary DataFrame with word index and 'count' column (weights)

    Returns:
        float: Weighted sentiment score
    """
    # Load volume word distribution
    df_vol = pd.read_csv(volume_path).set_index('word')
    total_words = df_vol['total_words'].max()

    # Join dictionary with volume (left join, add suffix to volume columns)
    df_joined = dict_df.join(df_vol, rsuffix='_data', how='left').fillna(0)

    # Multiply weight × word count
    df_joined['count_x_weight'] = df_joined['count'].multiply(df_joined['count_data'])

    # Sum and normalize by total words
    score = df_joined['count_x_weight'].sum() / total_words

    return score


def score_all_volumes(DF_ids, simple_dicts, weighted_dicts):
    """
    Score all volumes using all dictionaries and generate results DataFrame.

    Args:
        DF_ids: DataFrame with volume file paths
        simple_dicts: Dictionary of simple (unweighted) dictionaries
        weighted_dicts: Dictionary of weighted dictionaries

    Returns:
        DataFrame: Results with filename as index and score columns
    """
    # Initialize results DataFrame with filenames as index and all columns
    filenames = DF_ids['Filename'].tolist()

    # Create column names list (all dictionaries)
    all_dict_names = list(simple_dicts.keys()) + list(weighted_dicts.keys())

    # Initialize results with NaN
    results = pd.DataFrame(index=filenames, columns=all_dict_names, dtype=float)
    results.index.name = 'Filename'

    print("\n" + "="*60)
    print("SCORING ALL VOLUMES")
    print("="*60)
    print(f"Total volumes: {len(DF_ids):,}")
    print(f"Simple dictionaries: {len(simple_dicts)}")
    print(f"Weighted dictionaries: {len(weighted_dicts)}")
    print(f"Total metrics: {len(all_dict_names)}")
    print("="*60)

    print("\nProcessing all volumes...")
    for _, row in tqdm(DF_ids.iterrows(), total=len(DF_ids), desc="Volumes"):
        filename = row['Filename']
        volume_path = row['Path']

        # Load volume word distribution 
        df_vol = pd.read_csv(volume_path).set_index('word')
        total_words = df_vol['total_words'].max()

        # Score against all simple dictionaries
        for dict_name, dict_df in simple_dicts.items():
            # Join and calculate score 
            df_joined = dict_df.join(df_vol, how='left').fillna(0)
            score = df_joined['pct'].sum()
            results.loc[filename, dict_name] = score

        # Score against all weighted dictionaries
        for dict_name, dict_df in weighted_dicts.items():
            # Join and calculate score 
            df_joined = dict_df.join(df_vol, rsuffix='_data', how='left').fillna(0)
            df_joined['count_x_weight'] = df_joined['count'].multiply(df_joined['count_data'])
            score = df_joined['count_x_weight'].sum() / total_words
            results.loc[filename, dict_name] = score

    print(f"\nScoring complete! Generated {len(results)} rows × {len(results.columns)} columns")

    return results


def generate_word_distribution(raw_text_path, output_path):
    """
    Generate word distribution file from raw text.

    1. Read raw text
    2. Split into words
    3. Count occurrences
    4. Filter words appearing more than once
    5. Calculate percentages

    Args:
        raw_text_path: Path to raw cleaned text file
        output_path: Path to save word distribution CSV

    Returns:
        DataFrame: Word distribution with columns: word (index), count, pct, total_words
    """
    # Read raw text file
    with open(raw_text_path, 'rb') as f:
        raw = f.read().decode('utf-8')

    # Split into words
    lines = raw.split()

    # Create DataFrame and count words
    df = pd.DataFrame(lines, columns=['word'])
    df['count'] = 1
    df = df.groupby('word').sum().sort_values('count', ascending=False)

    # Filter words that appear more than once (matching notebook logic)
    df = df[df['count'] > 1]

    # Calculate percentage
    df['pct'] = df['count'] / df['count'].sum()

    # Add total_words column
    df['total_words'] = df['count'].sum()

    # Save to CSV
    df.to_csv(output_path)

    return df


def generate_all_distributions(source_dir, output_dir, filenames):
    """
    Generate word distributions for multiple volumes.

    Args:
        source_dir: Directory containing raw cleaned text files
        output_dir: Directory to save word distribution files
        filenames: List of filenames to process

    Returns:
        int: Number of files successfully generated
    """
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = []

    print("\n" + "="*60)
    print("GENERATING WORD DISTRIBUTIONS FROM RAW TEXT")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Files to process: {len(filenames)}\n")

    for filename in tqdm(filenames, desc="Generating distributions"):
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            if not os.path.exists(source_path):
                failed.append((filename, "Source file not found"))
                continue

            # Skip if output already exists
            if os.path.exists(output_path):
                successful += 1
                continue

            df = generate_word_distribution(source_path, output_path)
            successful += 1

        except Exception as e:
            failed.append((filename, str(e)))

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Successful: {successful}/{len(filenames)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for fname, error in failed[:5]:
            print(f"    - {fname}: {error}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")
    print("="*60)

    return successful


def get_prob_df():
    """
    Load index of all volume word distribution files.

    Returns:
        DataFrame with columns: HTID, Filename, Path (indexed by HTID)
    """
    path = r'./word_distributions'
    U = []
    # r=root, d=directories, f = files
    for r, d, f in tqdm(os.walk(path), desc = 'get_EF_htids'):
        for file in f:
            if '.txt' in file:
                htid = file.replace(".json.bz2","").replace("+",":").replace(",",".").replace("=", "/")
                filename = file
                U.append([htid, filename, r + '/' + filename])
    UK_files = pd.DataFrame(U, columns = ["HTID", "Filename", "Path"]).set_index('HTID')
    del U
    return UK_files


def save_final_analysis_format(results, output_dir='./output_final_analysis_format'):
    """
    Split unified results into 6 separate CSV files.

    Args:
        results: DataFrame with all sentiment scores (output from score_all_volumes)
        output_dir: Directory to save the 6 output files 

    Output files:
        1. Sentiment_scores_other.csv - Progress, Optimism, Pessimism, Regression
        2. progress_scores_main.csv - Main, Progress
        3. Sentiment_ChatGPT.csv - ChatGPT Progress 
        4. Optimism_abbr_industry_1708.csv - Industrialization_Prior, Optimism_Double_Meaning
        5. Industrialization_1643.csv - 1643 Dictionary scores 
        6. Industrialization_appleby.csv - Appleby Dictionary scores 
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SAVING FILES IN FINAL_ANALYSIS FORMAT")
    print("="*60)
    print(f"Output directory: {output_dir}\n")

    # 1. Sentiment_scores_other.csv
    file1 = os.path.join(output_dir, 'Sentiment_scores_other.csv')
    results[['Progress', 'Optimism', 'Pessimism', 'Regression']].to_csv(file1)
    print(f"[OK] Saved: Sentiment_scores_other.csv")

    # 2. progress_scores_main.csv (rename Secondary → Progress)
    df2 = results[['Main', 'Secondary']].copy()
    df2.rename(columns={'Secondary': 'Progress'}, inplace=True)
    file2 = os.path.join(output_dir, 'progress_scores_main.csv')
    df2.to_csv(file2)
    print(f"[OK] Saved: progress_scores_main.csv")

    # 3. Sentiment_ChatGPT.csv (rename column)
    df3 = results[['ChatGPT_Progress']].copy()
    df3.rename(columns={'ChatGPT_Progress': 'ChatGPT Progress'}, inplace=True)
    file3 = os.path.join(output_dir, 'Sentiment_ChatGPT.csv')
    df3.to_csv(file3)
    print(f"[OK] Saved: Sentiment_ChatGPT.csv")

    # 4. Optimism_abbr_industry_1708.csv
    file4 = os.path.join(output_dir, 'Optimism_abbr_industry_1708.csv')
    results[['Industrialization_Prior', 'Optimism_Double_Meaning']].to_csv(file4)
    print(f"[OK] Saved: Optimism_abbr_industry_1708.csv")

    # 5. Industrialization_1643.csv (1643 Dictionary scores)
    df5 = results[['Dict_1643']].copy()
    df5.rename(columns={'Dict_1643': '1643 Dictionary'}, inplace=True)
    file5 = os.path.join(output_dir, 'Industrialization_1643.csv')
    df5.to_csv(file5)
    print(f"[OK] Saved: Industrialization_1643.csv")

    # 6. Industrialization_appleby.csv (Appleby Dictionary scores)
    df6 = results[['Appleby']].copy()
    df6.rename(columns={'Appleby': 'Appleby Dictionary'}, inplace=True)
    file6 = os.path.join(output_dir, 'Industrialization_appleby.csv')
    df6.to_csv(file6)
    print(f"[OK] Saved: Industrialization_appleby.csv")

    print(f"\n{'='*60}")
    print(f"Successfully saved 6 files to: {output_dir}")
    print(f"These files can be copied to final_analysis/data/final_analysis_input/")
    print("="*60)

    return output_dir


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION
    # ========================================
    RAW_TEXT_SOURCE = r'F:\Claude\mallet-replication\Full_Deployment_Output'
    WORD_DIST_OUTPUT = r'./word_distributions'
    OUTPUT_DIR = './output'

    print("="*60)
    print("SENTIMENT SCORER - FULL DATASET")
    print("="*60)
    print(f"Raw text source:    {RAW_TEXT_SOURCE}")
    print(f"Word dist output:   {WORD_DIST_OUTPUT}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print("="*60)

    # ========================================
    # STEP 1: Get list of all files
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: Scanning raw text directory")
    print("="*60)

    all_filenames = [f for f in os.listdir(RAW_TEXT_SOURCE) if f.endswith('.txt')]

    print(f"Found {len(all_filenames)} text files")
    print(f"First file: {all_filenames[0]}")
    print(f"Last file:  {all_filenames[-1]}")

    # ========================================
    # STEP 2: Generate word distributions
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: Generating word distributions from raw text")
    print("="*60)

    num_generated = generate_all_distributions(
        source_dir=RAW_TEXT_SOURCE,
        output_dir=WORD_DIST_OUTPUT,
        filenames=all_filenames
    )

    print(f"\nSuccessfully generated {num_generated}/{len(all_filenames)} word distributions")

    # ========================================
    # STEP 3: Load dictionaries
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: Loading sentiment dictionaries")
    print("="*60)

    simple_dicts, weighted_dicts = load_dictionaries()
    print(f"Loaded {len(simple_dicts)} simple dictionaries and {len(weighted_dicts)} weighted dictionaries")

    # ========================================
    # STEP 4: Create volume index
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: Creating volume index")
    print("="*60)

    # Create volume index from generated word distributions
    volume_data = []
    for filename in all_filenames:
        htid = filename.replace(".json.bz2", "").replace("+", ":").replace(",", ".").replace("=", "/")
        path = os.path.join(WORD_DIST_OUTPUT, filename)
        volume_data.append([htid, filename, path])

    DF_ids = pd.DataFrame(volume_data, columns=["HTID", "Filename", "Path"]).set_index('HTID')
    print(f"Created index for {len(DF_ids)} volumes")

    # ========================================
    # STEP 5: Calculate sentiment scores
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: Calculating sentiment scores")
    print("="*60)

    results = score_all_volumes(DF_ids, simple_dicts, weighted_dicts)

    # ========================================
    # STEP 6: Save outputs
    # ========================================
    print("\n" + "="*60)
    print("STEP 6: Saving outputs")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save in final_analysis format (6 separate files)
    save_final_analysis_format(results, output_dir=OUTPUT_DIR)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Files processed:     {len(all_filenames)}")
    print(f"Word distributions:  {num_generated} generated")
    print(f"Sentiment scores:    {len(results)} rows x {len(results.columns)} columns")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  6 CSV files:")
    print(f"    - Sentiment_scores_other.csv")
    print(f"    - progress_scores_main.csv")
    print(f"    - Sentiment_ChatGPT.csv")
    print(f"    - Optimism_abbr_industry_1708.csv")
    print(f"    - Industrialization_1643.csv")
    print(f"    - Industrialization_appleby.csv")
    print("="*60)
