import pandas as pd
import pickle
import gc
import re
from functools import reduce
from src.utils import make_dir
from src.constants import manual_and_related_words


def clean_htids_topic_numbers(data, string_identifier):
    str_length = len(string_identifier)

    df = data.copy()
    df.drop(columns=0, inplace=True)
    df[1] = df[1].str.split('/').str[-1].str.strip("'")
    df.columns = ['HTID'] + [i for i in range(1,len(df.columns))]
    df['HTID'] = df['HTID'].str.replace('.txt', '')
    return df

def fix_htid(row):
    return row['HTID'].replace(":","+").replace("/", "=")

def flag_words(df, colnames, words, new_colname):
    """
    Flags words in a given column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    colname (str): The name of the column to search for words.
    words (list): A list of words to flag.
    new_colname (str): The name of the new column to create for flags.

    Returns:
    pd.DataFrame: The modified DataFrame with the new flag column.
    """
    pattern = '|'.join(map(re.escape, words))

    match_series = pd.Series(False, index=df.index)

    for colname in colnames:
        if colname not in df.columns:
            raise ValueError(f"Column '{colname}' not found in DataFrame.")
        # Create a new column with flags based on the pattern
        match_series |= df[colname].astype(str).str.contains(pattern, case=False, na=False)

    df[new_colname] = match_series.astype(int)  # Convert boolean to int (0 or 1)

    return df

def run_clean_data(config):

    #make directories if they don't exist
    make_dir(config['temporary_path'])
    make_dir(config['output_path'])

    print('Importing Data')
    if config['version'] == 'main_analysis':
        topic_data = pd.read_csv(config['input_path'] + 'topics.txt', sep = '\t', header = None)
        topic_keys = pd.read_csv(config['input_path'] + 'keys.txt', sep = '\t', header=None)
    elif config['version'] == 'coherence':
        topic_data = pd.read_csv(config['input_path'] + '40_Coherence_topics.txt', sep = '\t', header = None)
        topic_keys = pd.read_csv(config['input_path'] + '40_Coherence_keys.txt', sep = '\t', header=None)
    else:
        raise ValueError("Invalid version specified. Please choose 'main_analysis' or 'coherence'")
    

    metadata = pd.read_csv(config['input_path'] + 'metadata_march25.csv')
    sentiment = pd.read_csv(config['input_path'] + 'Sentiment_scores_other.csv')
    updated_progress = pd.read_csv(config['input_path'] + 'progress_scores_main.csv')
    industry = pd.read_csv(config['input_path'] + 'Industrialization_1643.csv')
    industry_scores_full_dict = pd.read_csv(config['input_path'] + 'Industrialization_appleby.csv')
    updated_optimism_industry = pd.read_csv(config['input_path'] + 'Optimism_abbr_industry_1708.csv')
    progress_chatgpt = pd.read_csv(config['input_path'] + 'Sentiment_ChatGPT.csv')


    print('Volume Data Dimensions:' + str(topic_data.shape))
    topic_data_cleaned = clean_htids_topic_numbers(data = topic_data, string_identifier='/Cleaned_Nov2024/')
    print('Volume Data Cleaned Dimensions:' + str(topic_data_cleaned.shape))

    #fix topic numbers
    topic_keys.drop(columns=0, inplace=True)
    topic_keys['topic_number'] = list(range(1,len(topic_keys)+1))
    topic_keys.columns = ['weight', 'words', 'topic_number']

    metadata = metadata.rename(columns={'Unnamed: 0': 'HTID', 'year': 'Year'})
    metadata['Year_rounded'] = pd.to_numeric(metadata['Year'])
    metadata['Year'] = pd.to_numeric(metadata['Year'], downcast='signed')
    metadata['HTID'] = metadata.apply(fix_htid, axis=1)
    translations = pd.read_csv(config['input_path'] + 'translations.csv')
    translations['HTID'] = translations.apply(fix_htid, axis=1)
    metadata = metadata.merge(translations, on= 'HTID', how = 'left')

    #flag manual and related words
    metadata = flag_words(df = metadata, colnames=['title_translations', 'description'], words=manual_and_related_words, new_colname='manual_flag')
    metadata = metadata.drop(columns=['title_translations', 'description'], axis=1)

    sentiment = sentiment.rename(columns = {'Filename': 'HTID', 'Regression': 'percent_regression', 'Pessimism': 'percent_pessimism', 'Optimism':'percent_optimistic', 'Progress': 'percent_progress_original'})
    sentiment['HTID'] = sentiment['HTID'].map(lambda x: x.replace('.txt', '')) #remove '.txt' at the end of each string for HTIDs

    updated_progress = updated_progress.rename(columns={'Filename': 'HTID', 'Main': 'percent_progress_main', 'Progress': 'percent_progress_secondary'})
    updated_progress['HTID'] = updated_progress['HTID'].map(lambda x: x.replace('.txt', ''))

    industry = industry.rename(columns={'Filename': 'HTID', '1643 Dictionary':'industry'})
    industry['HTID'] = industry['HTID'].map(lambda x: x.replace('.txt', ''))#remove '.txt' at the end of each string for HTIDs

    industry_scores_full_dict = industry_scores_full_dict.rename(columns={'Filename': 'HTID', 'Appleby Dictionary':'industry_full_dict'})
    industry_scores_full_dict['HTID'] = industry_scores_full_dict['HTID'].map(lambda x: x.replace('.txt', ''))#remove '.txt' at the end of each string for HTIDs

    updated_optimism_industry = updated_optimism_industry.rename(columns={'Filename': 'HTID', 'Optimism_Double_Meaning':'optimism_abbreviated', 'Industrialization_Prior': 'industry_1708'})
    updated_optimism_industry['HTID'] = updated_optimism_industry['HTID'].map(lambda x: x.replace('.txt', ''))#remove '.txt' at the end of each string for HTIDs

    progress_chatgpt = progress_chatgpt.rename(columns={'Filename': 'HTID', 'ChatGPT Progress': 'progress_chatgpt'})
    progress_chatgpt['HTID'] = progress_chatgpt['HTID'].map(lambda x: x.replace('.txt', ''))#remove '.txt' at the end of each string for HTIDs

    sentiment_dfs = [industry, industry_scores_full_dict, sentiment, updated_progress, updated_optimism_industry, progress_chatgpt]

    print('Sentiment Dimensions:' + str(sentiment.shape))
    print('Industry Dimensions:' + str(industry.shape))
    print('Updated Progress Dimensions:' + str(updated_progress.shape))

    sentiment_scores_all = reduce(lambda left,right: pd.merge(left, right, on = 'HTID', how = 'inner'), sentiment_dfs) #merge on volume ID

    sentiment_scores_all.fillna(0, inplace=True) #fill NA values with 0, for a few novels without words.

    print('Merged Dimensions:' + str(sentiment_scores_all.shape))

    print('Exporting Data')
    topic_data_cleaned.to_csv(config['temporary_path'] + 'topic_weights.csv', index = False)
    topic_keys.to_csv(config['temporary_path'] + 'topics.csv', index = False)
    metadata.to_csv(config['temporary_path'] + 'metadata.csv', index=False)
    sentiment_scores_all.to_csv(config['temporary_path'] + 'sentiment_scores.csv', index=False)

    del topic_data, topic_keys, metadata, topic_data_cleaned, sentiment, updated_progress, industry, sentiment_scores_all, industry_scores_full_dict, updated_optimism_industry, progress_chatgpt
    gc.collect()