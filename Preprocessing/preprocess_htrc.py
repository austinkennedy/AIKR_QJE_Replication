#!/usr/bin/env python3
"""
HTRC Text Preprocessing for Topic Modeling - Replication Script

This script preprocesses HTRC Extracted Features files to produce clean text
suitable for topic modeling with MALLET.

Version: 2.1
Last Updated: 2025-11-01
"""

import os
import sys
import unicodedata
import pycountry
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import logging
from htrc_features import FeatureReader
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import names, stopwords
from nltk.corpus import words as nltk_words
import argparse
import re


# POS tags to retain (standard practice for topic modeling)
POS_TAGS = ('NE', 'NN', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR',
            'IN', 'DT', 'VB', 'VBP', 'VBZ', 'VBD', 'VBN', 'VBG',
            'RB', 'RBR', 'RBS', 'RP', 'CC')

# Minimum word length (characters)
MIN_WORD_LENGTH = 2

# Minimum word frequency per volume
MIN_WORD_FREQUENCY = 2

# Character corrections (OCR error handling)
GREEK_CORRECTION = str.maketrans('ºªſβ', 'oasb')

# Ligatures mapping
LIGATURES = {
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬅ': 'ft',
    'ﬃ': 'ffi',
    'ﬀ': 'ff',
    'ﬄ': 'ffl'
}

# Stopword categories
# Only english_stopwords and roman_numerals are filtered
# All other categories are used only for stem validation, not filtering
STOPWORD_FILTERS = {
    'cities': False,
    'countries': False,
    'people_names': False,
    'english_stopwords': True,
    'modern_words': False,
    'continents': False,
    'days_months': False,
    'roman_numerals': True,
    'stems': False
}

# ============================================================================


# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
idx = pd.IndexSlice

# Character deletion (remove all non-alphabetic characters)
non_alpha_chars = ''.join(c for c in map(chr, range(256)) if not c.isalpha())
non_alpha_translator = str.maketrans('', '', non_alpha_chars)


# ============================================================================
# MODULE-LEVEL DICTIONARY LOADING
# ============================================================================
# These dictionaries are loaded when the module is imported (not at runtime).
# This ensures each multiprocessing worker gets its own copy of the data.
# Dictionaries are loaded from the reference_data/ subdirectory.
# ============================================================================

# Helper function for Roman numerals (needed before loading)
def int_to_roman_lowercase(num):
    """Convert integer to lowercase Roman numeral"""
    if num == 0:
        return "n"  # Special case for zero (nulla)

    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "m", "cm", "d", "cd",
        "c", "xc", "l", "xl",
        "x", "ix", "v", "iv",
        "i"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


# Default dictionary paths (relative to this script)
DEFAULT_DICT_CORRECTIONS = Path(__file__).parent / "reference_data" / "Master_Corrections.csv"
DEFAULT_DICT_MA = Path(__file__).parent / "reference_data" / "MA_Dict_Final.csv"
DEFAULT_WORLD_CITIES = Path(__file__).parent / "reference_data" / "world_cities.csv"

# Load spelling corrections dictionaries
spelling_corrections = pd.read_csv(DEFAULT_DICT_CORRECTIONS)
spelling_corrections = spelling_corrections.rename(columns={'lemma':'orig','correct_spelling':'stand'}).set_index('orig')
spelling_corrections_index = set(spelling_corrections.index)

archaic_to_modern_dict = pd.read_csv(DEFAULT_DICT_MA)
archaic_to_modern_dict = archaic_to_modern_dict.set_index('orig')
archaic_words_index = set(archaic_to_modern_dict.index)

# Load geographic data
cities_df = pd.read_csv(DEFAULT_WORLD_CITIES)
cities = set(city.lower() for city in cities_df['name'])

countries = set([country.name.lower() for country in pycountry.countries])

continents = set(['africa', 'asia', 'europe', 'america', 'australia', 'antartica'])

# Load NLTK data
people_names = set([name.lower() for name in names.words()])
english_stopwords = set(stopwords.words('english'))
modern_words = set([word.lower() for word in nltk_words.words()])

# Generate stems
stems = set([stemmer.stem(word) for word in modern_words])

# Days and months
days = set(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
months = set(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
             'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

# Generate Roman numerals (0-500)
roman_numerals = set([int_to_roman_lowercase(i) for i in range(501)])

# Stopword sets with different purposes:
# 1. stem_validation_dict: Large reference dictionary (~500k terms) used in lemmatize_or_stem()
#    to validate whether a stemmed form is a legitimate word 
stem_validation_dict = cities.union(
    countries, people_names, english_stopwords, modern_words,
    continents, stems, days, months, roman_numerals
)
# 2. filtered_stopwords: Words to actually remove from output based on STOPWORD_FILTERS configuration
#    Only 2 categories are filtered: english_stopwords + roman_numerals (~680 terms)
filtered_stopwords = set()
if STOPWORD_FILTERS.get('cities', False):
    filtered_stopwords = filtered_stopwords.union(cities)
if STOPWORD_FILTERS.get('countries', False):
    filtered_stopwords = filtered_stopwords.union(countries)
if STOPWORD_FILTERS.get('people_names', False):
    filtered_stopwords = filtered_stopwords.union(people_names)
if STOPWORD_FILTERS.get('english_stopwords', False):
    filtered_stopwords = filtered_stopwords.union(english_stopwords)
if STOPWORD_FILTERS.get('modern_words', False):
    filtered_stopwords = filtered_stopwords.union(modern_words)
if STOPWORD_FILTERS.get('continents', False):
    filtered_stopwords = filtered_stopwords.union(continents)
if STOPWORD_FILTERS.get('days_months', False):
    filtered_stopwords = filtered_stopwords.union(days, months)
if STOPWORD_FILTERS.get('roman_numerals', False):
    filtered_stopwords = filtered_stopwords.union(roman_numerals)
if STOPWORD_FILTERS.get('stems', False):
    filtered_stopwords = filtered_stopwords.union(stems)

# ============================================================================


def parse_config_file(config_path):
    """Parse shell-style config file"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Match VAR="value" or VAR='value' or VAR=value
                match = re.match(r'^([A-Z_]+)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    # Remove quotes if present
                    value = value.strip('"\'')
                    # Skip if empty (unless it's meant to be empty)
                    if value:
                        config[key] = value
    except Exception as e:
        logging.error(f"Error parsing config file {config_path}: {e}")
        sys.exit(1)
    return config


def parse_arguments():
    """Parse command-line arguments and configuration file"""
    parser = argparse.ArgumentParser(
        description='HTRC Text Preprocessing for Topic Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''

CONFIGURATION:
  Settings can be provided via config file OR command-line arguments.

  Option 1 - Use config file (recommended):
    1. cp config.template.sh config.sh
    2. Edit config.sh with your paths
    3. python preprocess_htrc.py --config config.sh

  Option 2 - Use command-line arguments:
    python preprocess_htrc.py --input <path> --output <path> [options]

EXAMPLES:
  # Using config file (recommended):
  python preprocess_htrc.py --config config.sh

  # Using command-line arguments:
  python preprocess_htrc.py \\
      --input /path/to/htrc_files \\
      --output /path/to/cleaned_text \\
      --dict-corrections ./reference_data/Master_Corrections.csv \\
      --dict-ma ./reference_data/MA_Dict_Final.csv \\
      --world-cities ./reference_data/world_cities.csv

  # Preview without running:
  python preprocess_htrc.py --config config.sh --dry-run

For detailed documentation, see README.md
        '''
    )

    # Config file
    parser.add_argument('--config', type=Path,
                       help='Configuration file (shell format)')

    # Required paths
    parser.add_argument('--input', '--input-dir', type=Path, dest='input',
                       help='Input directory with HTRC .json.bz2 files')
    parser.add_argument('--output', '--output-dir', type=Path, dest='output',
                       help='Output directory for cleaned text files')
    parser.add_argument('--dict-corrections', type=Path, dest='dict_corrections',
                       help='Path to Master_Corrections.csv')
    parser.add_argument('--dict-ma', type=Path, dest='dict_ma',
                       help='Path to MA_Dict_Final.csv')
    parser.add_argument('--world-cities', type=Path, dest='world_cities',
                       help='Path to world_cities.csv')

    # Optional arguments
    parser.add_argument('--num-processes', type=int, dest='num_processes',
                       help='Number of CPU processes (default: auto-detect)')
    parser.add_argument('--error-log', type=Path, dest='error_log',
                       help='Error log file path (optional)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load config file if specified
    if args.config:
        if not args.config.exists():
            parser.error(f"Config file not found: {args.config}")
        config = parse_config_file(args.config)

        # Apply config values if not overridden by CLI
        if not args.input and 'INPUT_DIR' in config:
            args.input = Path(config['INPUT_DIR'])
        if not args.output and 'OUTPUT_DIR' in config:
            args.output = Path(config['OUTPUT_DIR'])
        if not args.dict_corrections and 'DICT_CORRECTIONS' in config:
            args.dict_corrections = Path(config['DICT_CORRECTIONS'])
        if not args.dict_ma and 'DICT_MA' in config:
            args.dict_ma = Path(config['DICT_MA'])
        if not args.world_cities and 'WORLD_CITIES' in config:
            args.world_cities = Path(config['WORLD_CITIES'])
        if not args.num_processes and 'NUM_PROCESSES' in config:
            args.num_processes = int(config['NUM_PROCESSES'])
        if not args.error_log and 'ERROR_LOG' in config:
            args.error_log = Path(config['ERROR_LOG'])


    required = {
        'input': args.input,
        'output': args.output
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(
            f"Missing required arguments: {', '.join(missing)}\n\n"
            "Provide them via config file (--config config.sh) or command-line arguments.\n"
            "Run with --help for more information."
        )

    return args


def setup_logging(args):
    """Configure logging with optional file output"""
    handlers = [logging.StreamHandler(sys.stderr)]

    if args.error_log:
        handlers.append(logging.FileHandler(args.error_log))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_environment(args):
    """Validate environment and dependencies before processing"""
    errors = []

    print("Validating environment...")

    # Check input directory exists and has files
    if not args.input.exists():
        errors.append(f"Input directory does not exist: {args.input}")
    else:
        # Check for .json.bz2 files
        json_files = list(args.input.rglob('*.json.bz2'))
        if not json_files:
            errors.append(f"No .json.bz2 files found in: {args.input}")
        else:
            print(f"  [OK] Found {len(json_files)} .json.bz2 files in input directory")

    # Check output directory
    if args.output.exists() and not args.dry_run:
        print(f"  [WARNING] Output directory exists: {args.output}")
        print("    Files may be overwritten! Continuing anyway...")

    # Check dictionary files exist (loaded from default paths at module import)
    print(f"  [OK] Dictionary files loaded from default paths (reference_data/)")

    # Check Python libraries
    try:
        import htrc_features
        import pandas
        import pycountry
        import tqdm
        import nltk
        print("  [OK] All required Python libraries are installed")
    except ImportError as e:
        errors.append(f"Required library not installed: {e}")

    # Check NLTK data
    nltk_data_required = ['words', 'names', 'stopwords', 'wordnet']
    missing_nltk = []

    for dataset in nltk_data_required:
        try:
            nltk.data.find(f'corpora/{dataset}')
        except LookupError:
            missing_nltk.append(dataset)

    if missing_nltk:
        errors.append(
            f"NLTK data not found: {', '.join(missing_nltk)}\n"
            f"    Install with: python -m nltk.downloader {' '.join(missing_nltk)}"
        )
    else:
        print("  [OK] All required NLTK data is available")

    if errors:
        print("\n" + "="*80)
        print("ERROR: Environment validation failed")
        print("="*80)
        for error in errors:
            print(f"\n[ERROR] {error}")
        print("\nPlease fix these issues and try again.")
        print("="*80)
        sys.exit(1)

    print("  [OK] Environment validation passed")


def print_configuration(args):
    """Display configuration for normalize_and_clean_wordparency"""
    print("\n" + "="*80)
    print("HTRC Preprocessing")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Input Directory:      {args.input}")
    print(f"  Output Directory:     {args.output}")
    print(f"\n  Dictionary Files:")
    print(f"    Corrections:        {args.dict_corrections}")
    print(f"    Modern/Archaic:     {args.dict_ma}")
    print(f"    World Cities:       {args.world_cities}")

    if args.num_processes:
        print(f"\n  CPU Processes:        {args.num_processes} (user-specified)")
    else:
        print(f"\n  CPU Processes:        {mp.cpu_count()} (auto-detected)")

    if args.error_log:
        print(f"  Error Log:            {args.error_log}")

    print("\nProcessing Parameters (FIXED FOR REPLICATION):")
    print(f"  POS Tags:             {len(POS_TAGS)} tags")
    print(f"  Min Word Length:      {MIN_WORD_LENGTH} characters")
    print(f"  Min Word Frequency:   {MIN_WORD_FREQUENCY} per volume")
    enabled_filters = sum(1 for v in STOPWORD_FILTERS.values() if v)
    print(f"  Stopword Filters:     {enabled_filters} of {len(STOPWORD_FILTERS)} categories enabled (english_stopwords + roman_numerals)")
    print("="*80)
    print()


def validate_reference_data(args):
    """Validate that reference data was loaded successfully at module import time"""
    print("Validating reference data...")

    try:
        # Check that dictionaries were loaded (they load at import time)
        if spelling_corrections is None or len(spelling_corrections) == 0:
            logging.error("Spelling corrections not loaded")
            sys.exit(1)
        print(f"  [OK] Loaded {len(spelling_corrections)} spelling corrections")

        if archaic_to_modern_dict is None or len(archaic_to_modern_dict) == 0:
            logging.error("Modern/archaic mappings not loaded")
            sys.exit(1)
        print(f"  [OK] Loaded {len(archaic_to_modern_dict)} modern/archaic mappings")

        print(f"  [OK] Loaded {len(cities)} city names")
        print(f"  [OK] Loaded {len(countries)} countries")
        print(f"  [OK] Loaded {len(people_names)} people names")
        print(f"  [OK] Loaded {len(english_stopwords)} English stopwords")
        print(f"  [OK] Loaded {len(modern_words)} modern words")
        print(f"  [OK] Reference dictionary for stem validation: {len(stem_validation_dict)} terms")
        enabled_filters = sum(1 for v in STOPWORD_FILTERS.values() if v)
        print(f"  [OK] Stopwords filtered from output: {len(filtered_stopwords)} terms ({enabled_filters} of {len(STOPWORD_FILTERS)} categories: english_stopwords + roman_numerals)")

    except Exception as e:
        logging.error(f"Error validating reference data: {e}")
        sys.exit(1)


def scan_htrc_files(input_path: Path) -> pd.DataFrame:
    """
    Generate a DataFrame of HTRC Extracted Features files.

    Args:
        input_path: Directory containing .json.bz2 files

    Returns:
        pd.DataFrame: A DataFrame with columns 'HTID', 'Filename', and 'Path', indexed by 'HTID'.
    """
    def generate_files():
        total_files = sum(len(files) for _, _, files in os.walk(input_path))
        print(f"Scanning: {input_path}")
        with tqdm(total=total_files, desc="Scanning files", unit=" files") as pbar:
            for r, d, f in os.walk(input_path):
                for file in f:
                    if '.json.bz2' in file:
                        htid = file.replace(".json.bz2","").replace("+",":").replace(",",".").replace("=", "/")
                        yield [htid, file, r]
                    pbar.update(1)

    htrc_files = pd.DataFrame(generate_files(), columns=["HTID", "Filename", "Path"]).set_index('HTID')
    return htrc_files


def getFeatureReader(final_ids):
    """Create FeatureReader from file paths"""
    # Build file paths using vectorized pandas operations 
    file_paths = list(final_ids['Path'] + os.sep + final_ids['Filename'])
    corpus = FeatureReader(file_paths)
    return corpus


def lemmatize_or_stem(cleaned):
    """Lemmatize or stem a word based on POS tag"""
    if cleaned[1].startswith('N'):
        lemmatized = lemmatizer.lemmatize(cleaned[0], 'n')
    elif cleaned[1].startswith('V'):
        lemmatized = lemmatizer.lemmatize(cleaned[0], 'v')
    elif cleaned[1].startswith('J'):
        lemmatized = lemmatizer.lemmatize(cleaned[0], 'a')
    elif cleaned[1].startswith('R'):
        lemmatized = lemmatizer.lemmatize(cleaned[0], 'r')
    else:
        lemmatized = lemmatizer.lemmatize(cleaned[0])

    if lemmatized == cleaned[0]:
        lemmatized = lemmatizer.lemmatize(cleaned[0])  # Try without POS

    if lemmatized == cleaned[0]:
        stem = stemmer.stem(cleaned[0])  # Try stemming
        if stem in stem_validation_dict:
            return stem

    return lemmatized


def normalize_and_clean_word(row):
    """Transform a word: remove punctuation, fix Greek chars, normalize Unicode"""
    string = row.name[0]
    string = string.translate(non_alpha_translator)
    string = string.translate(GREEK_CORRECTION)

    # Fix ligatures
    for ligature, replacement in LIGATURES.items():
        string = string.replace(ligature, replacement)

    string = unicodedata.normalize('NFKC', string)
    return pd.Series({'corrected': string})


def clean_punctuation(words):
    """Clean punctuation and filter by minimum length"""
    words_copy = words.copy()
    words_copy[['corrected']] = words_copy.apply(normalize_and_clean_word, axis=1)
    words_copy = words_copy.reset_index().drop(columns='lowercase')
    return words_copy[words_copy['corrected'].map(len) > MIN_WORD_LENGTH].sort_values('count', ascending=False)


def ma_search(row):
    """Look up word in Modern/Archaic dictionary"""
    try:
        if row in archaic_words_index:
            selection = archaic_to_modern_dict.loc[row]
            return selection.stand
        else:
            return row
    except Exception as e:
        logging.error(f"Error in ma_search: {e}")
        return 'error'


def stem(row):
    """Stem a word"""
    return stemmer.stem(row)


def spell_correction_lookup(row):
    """Look up word in spelling corrections dictionary"""
    try:
        if row in spelling_corrections_index:
            selection = spelling_corrections.loc[row]
            return selection.stand
        else:
            return row
    except Exception as e:
        logging.error(f"Error in spell_correction_lookup: {e}")
        return 'error'


def process_volume_pipeline(volume):
    """Process a volume: extract tokens, clean, lemmatize, filter"""
    token_list = volume.tokenlist(pages=False, case=False, section='body')
    token_list.index = token_list.index.droplevel(0)
    filtered_tokens = token_list[token_list.index.get_level_values(1).isin(POS_TAGS)]
    filtered_tokens = clean_punctuation(filtered_tokens)
    filtered_tokens = filtered_tokens.assign(corrected=filtered_tokens.corrected.map(spell_correction_lookup))
    filtered_tokens = filtered_tokens.groupby(['corrected','pos']).sum()
    filtered_tokens = filtered_tokens[filtered_tokens['count'] >= MIN_WORD_FREQUENCY]
    filtered_tokens = filtered_tokens.loc[~filtered_tokens.index.get_level_values('corrected').isin(filtered_stopwords)]
    filtered_tokens = filtered_tokens.assign(lemma=filtered_tokens.index.map(lemmatize_or_stem))
    filtered_tokens = filtered_tokens.groupby('lemma').sum()
    filtered_tokens = filtered_tokens.loc[~filtered_tokens.index.get_level_values('lemma').isin(filtered_stopwords)]
    words_without_archaic = filtered_tokens.loc[~filtered_tokens.index.get_level_values('lemma').isin(archaic_words_index)]
    words_with_archaic = filtered_tokens.loc[filtered_tokens.index.get_level_values('lemma').isin(archaic_words_index)]
    words_with_archaic = words_with_archaic.assign(corrected_ma=words_with_archaic.index.get_level_values('lemma').map(ma_search))
    words_with_archaic = words_with_archaic.groupby('corrected_ma').sum()
    words_with_archaic = words_with_archaic.loc[~words_with_archaic.index.get_level_values('corrected_ma').isin(filtered_stopwords)]
    combined_processed_words = pd.concat([words_without_archaic, words_with_archaic])
    combined_processed_words = combined_processed_words.assign(stem=combined_processed_words.index.map(stem))
    combined_processed_words = combined_processed_words.groupby('stem').sum()
    return combined_processed_words


def process_volume(volume, output_path):
    """Process a single volume and write to file"""
    save_path = output_path / f"{volume.id.replace(':','+').replace('/','=')}.txt"
    try:
        clean_df = process_volume_pipeline(volume)
        if clean_df is None:
            logging.warning(f"No clean data for volume {volume.id}")
            return None
        clean_df = clean_df.sort_values('count', ascending=False)
        with open(save_path, 'w', encoding='utf8') as output:
           for item, value in clean_df.iterrows():
               to_print = [(item + ' ') * int(value['count'])]
               output.write(str(to_print[0]))
        return clean_df
    except Exception as e:
        logging.error(f"Error processing volume {volume.id}: {e}")
        return None


def process_volume_wrapper(args_tuple):
    """Wrapper for multiprocessing"""
    volume, output_path = args_tuple
    return process_volume(volume, output_path)


def CleanAndWrite(corpus, output_path, num_processes=None, volume_limit=None):
    """Process all volumes using multiprocessing"""
    total_volumes = len(corpus)

    if num_processes is None:
        num_processes = mp.cpu_count()

    # Use generator to avoid loading all volumes into memory at once
    def volume_generator():
        count = 0
        for volume in corpus.volumes():
            if volume_limit is not None and count >= volume_limit:
                break
            yield (volume, output_path)
            count += 1

    # Adjust total if limiting
    if volume_limit is not None:
        print(f"Limiting processing to first {volume_limit} volumes (out of {total_volumes} total)")
        total_volumes = min(volume_limit, total_volumes)

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_volume_wrapper, volume_generator()),
            total=total_volumes,
            desc="Processing volumes"
        ))

    successful = sum(1 for r in results if r is not None)
    print(f"\nProcessed {successful}/{total_volumes} volumes successfully")


def main():
    """Main preprocessing pipeline"""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args)

    # Display configuration
    print_configuration(args)

    # Validate environment
    validate_environment(args)
    print()

    # Validate reference data 
    validate_reference_data(args)
    print()

    # Scan for HTRC files
    print("="*80)
    print("Step 1/3: Scanning for HTRC Extracted Features files")
    print("="*80)
    htrc_files = scan_htrc_files(args.input)
    print(f"Found {len(htrc_files)} HTRC Extracted Features files")
    print()

    if args.dry_run:
        print("="*80)
        print("DRY RUN MODE")
        print("="*80)
        print(f"\nWould process {len(htrc_files)} volumes")
        print(f"Output directory: {args.output}")
        print(f"Output format: One .txt file per volume")
        print("\nRemove --dry-run flag to execute processing.")
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {args.output}\n")

    # Initialize FeatureReader
    print("="*80)
    print("Step 2/3: Initializing HTRC FeatureReader")
    print("="*80)
    corpus = getFeatureReader(htrc_files)
    print(f"FeatureReader initialized with {len(corpus)} volumes")
    print()

    # Process and write volumes
    print("="*80)
    print("Step 3/3: Processing and writing cleaned text")
    print("="*80)
    CleanAndWrite(corpus, args.output, args.num_processes)

    # Success message
    print("\n" + "="*80)
    print("SUCCESS! Preprocessing complete")
    print("="*80)
    print(f"\nCleaned text files: {args.output}")
    print(f"Total volumes: {len(corpus)}")
    print("\nOutput files are ready for MALLET topic modeling.")
    print("="*80)


if __name__ == "__main__":
    mp.freeze_support()  # Necessary for Windows
    main()
