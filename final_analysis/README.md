# Final Analysis - Replication Scripts

This repository contains scripts that take the output of the LDA model, run the categorization algorithm, and produce final figures and tables.

## Requirements

This code was developed and run on Python 3.11.1. A `requirements.txt` file is provided listing dependencies and package versions. You can install it by navigating to the `final_analysis` directory in the command line and running:

```
pip install -r requirements.txt
```

## Quick Start

The data for this part of the analysis can be found in the `final_analysis_input` folder in our data repository: [Data For Replication](https://www.dropbox.com/scl/fo/2qg8lv11j41ytjp2ru3k7/AHfF5xuQUVdtFNYKjcwMBa0?rlkey=py6mt8kztk72g8ity4hqlpbqc&st=9d8zn45r&dl=0)

This data includes the output of Stages 1-3 of the pipeline as detailed in [`README.md`](README.md) in the parent directory.

Those wishing to replicate the final analysis should either run Stages 1-3 of the pipeline or download the `final_analysis_input` folder from the data repository, then create a directory named `data` within the `final_analysis` directory, and place the `final_analysis_input` folder and files into the `data` directory. Alternatively, one could store the data elsewhere and change the `input_path` parameters in the config files, as detailed below.

After this step is completed, the entirety of the final analysis can be run by running the `main.py` file in Python: 

**For Windows Users**
```
python main.py
```

**For MacOS/Linux Users**
```
python3 main.py
```

Note that all necessary input data (i.e. the output of the LDA model and additional files) must be present in order for the code to properly run. The code will create all figures and tables and store them in the `data/paper_assets/` directory.

## Data

The following is a table of data files provided in the `final_analysis_input` folder that must be present for the analysis to run:

| File | Description |
|------|-------------|
| `40_Coherence_keys.txt` | The topics and associated words outputted by the LDA model using the Coherence score (Section C). |
| `40_Coherence_topics.txt` | The volume-level weights associated with each topic, using the topics outputted by the LDA model using the Coherence score (Section C). |
| `estc_1500_to_1800.csv` | Data for the ESTC section (Section D). |
| `famous_books.csv` | The metadata for selected famous works in our corpus. |
| `Optimism_abbr_industry_1708.csv` | The volume-level optimism scores, double meanings excluded, and industry scores, using the 1708 dictionary. |
| `Industrialization_appleby.csv` | The volume-level industry scores, including words first used after 1643. |
| `Industrialization_1643.csv` | The volume-level industry scores, excluding words first used after 1643. |
| `keys.txt` | The topics and associated words outputted by the LDA model used in our main analysis. |
| `topics.txt` | The volume-level weights associated with each topic, using the topics outputted by the LDA model used in our main analysis. |
| `metadata_march25.csv` | The metadata for our corpus. |
| `Sentiment_ChatGPT.csv` | The volume-level scores for the progress dictionary produced by ChatGPT. |
| `Sentiment_scores_other.csv` | The sentiment scores for "Optimism", "Pessimism", "Regression", and the "Progress" dictionary including words first used after 1643 |
| `translations.csv` | Contains flags for volumes that are translations. |
| `progress_scores_main.csv` | Scores for the "progress" dictonaries used in the main analysis (excluding words after 1643), as well as the 1708 dictionary. |

Any files in the above list that are not created by Stages 1-3 can be downloaded from the `final_anlaysis_input` folder in the data repository.

## Scripts

### Structure
The files in the `final_analysis` directory are structured as follows, assuming that a `data` directory has been created and the `final_analysis_input` folder has been downloaded and placed inside:

```text
.
├── data/
│   └── final_analysis_input/
├── Rscripts/
│   ├── additional_ternary_figs.R
│   ├── famous_books.R
│   ├── install_dependencies.R
│   ├── marginal_predicted_figs.R
│   ├── r_config.yaml
│   └── regression_tables.R
├── configs/
│   ├── config_coherence.yaml
│   └── config_main_analysis.yaml
├── src/
│   ├── categories.py
│   ├── clean_data.py
│   ├── constants.py
│   ├── cross_topics.py
│   ├── figures.py
│   ├── shares.py
│   ├── topic_volume_weights.py
│   ├── utils.py
│   └── volume_data.py
├── .gitignore
├── README.md
├── coherence.py
├── main.py
├── main_analysis.py
├── requirements.txt
├── sync_assets.py
└── unbinned_analysis.py
```

### Orchestration Scripts

There are five ochestration scripts in the `final_analysis` directory.

`main.py` runs the entire analysis and produces all figures and tables, including for the appendix. The other orchestration scripts are called by `main.py`, but can be run individually as well:
- `main_analysis.py`: this script takes the output of the main LDA model and runs the algorithm to produce distinct categories (e.g. Science, Religion, and Political Economy in the paper), runs regressions, and produces the figures and tables for the main analysis.
- `unbinned_analysis.py`: this script re-runs the main analysis without placing volumes into bins.
- `coherence.py`: This script re-runs the analysis using the output of the LDA model which used the Coherence scores (Section C)
- `sync_assets.py`: This script retrieves all of the figures and tables produced by the other scripts and consolidates them into a single folder, with tables and figures corresponding to their respective labels in the paper.

### Source Code

The orchestration scripts call functions from scripts located in the `src/` directory. Below is a list of these scripts and a description of what each one does, in the order that the analysis runs:
- `clean_data.py`: imports input data, performs some basic data cleaning, and produces intermediate datasets.
- `cross_topics.py`: performs topic-wise multiplication on the volume topic weights.
- `categories.py`: runs the algorithm to create distinct categories.
- `shares.py`: calculates yearly shares for each cross-topic combination.
- `topic_volume_weights.py`: calculates category weights for each volume.
- `volume_data.py`: combine category weights, all sentiment scores, and calculate percentiles.
- `figures.py`: produces main figures.

As part of the analysis, the orchestration scripts run R scripts located in the `Rscripts/` directory. Below is a list of these scripts and a description for each one:
- `regression_tables.R`: runs regressions and produces regression tables.
- `marginal_predicted_figs.R`: runs regressions, calculates predicted values, and produces predicted values figures.
- `famous_books.R`: produces figure showing category weights for selected famous volumes.
- `additional_ternary_figs.R`: creates additional ternary plots, including bi-scale ternary plots and plots with sub-triangles.
- `install_dependencies.R`: installs R package dependencies.

### Configuration Files

Configuration files can be found in the `configs/` directory. There are two configuration files:
1. `config_main_analysis.yaml`: the configuration file for the main analysis.
2. `config_coherence.yaml`: the configuration file for the analysis using the Coherence score.

The files contain the following parameters:
- `version`: points the data cleaning script to the correct files and data cleaning steps.
- `input_path`: the location of the input data.
- `temporary_path` the location to save temporary data files.
- `output_path`: the location to save output tables and figures
- `categories`: determines the categories and associated topic numbers. **Note:** This parameter is set based on the results of the categorization algorithm.
- `half_century`: (true/false) determines whether yearly figures are created for every year in the corpus time period or only for half-centuries. Expect a significantly longer runtime if this parameter is set to false.
- `bins`: (true/false) whether or not to place volumes into bins of +/- 20 years. Changing this parameter to false runs the unbinned analysis.
- `category_plots_ymax`: sets the maximum y-value for the category plots.
- `min_regression_year`: sets the minimum year of publication for volumes included in regressions.
- `ternary_figs`: sets the options for the creation of ternary figures.

### Note on the Categorization Algorithm

Currently, the parameters for `categories` in the config files are pre-set to match those used in the paper, to allow the entire analysis to run by running `main.py`. To replicate the process of selecting categories in its entirety, one would need to take the raw output produced by `categories.py` and imitate the selection process that can be found in the `clusters_qje_r2.xlsx` file in the `final_analysis_input` folder in our [Data For Replication](https://www.dropbox.com/scl/fo/2qg8lv11j41ytjp2ru3k7/AHfF5xuQUVdtFNYKjcwMBa0?rlkey=py6mt8kztk72g8ity4hqlpbqc&st=9d8zn45r&dl=0) repository.

## List of Figures and Tables

Below is a list of figures and tables produced by this analysis, as well as their corresponding generating source code:

| Figure/Table # | Source Code | Output File (in `data/paper_assets/`) |
|---------------|---------|-------------|
| Figure 1 | `src/figures.py` | `fig_1/total_volumes_raw.png` |
| Figure 2 | `src/figures.py` | `fig_2/1550.png`<br>`fig_2/1600.png`<br>`fig_2/1650.png`<br>`fig_2/1700.png`<br>`fig_2/1750.png`<br>`fig_2/1800.png`<br>`fig_2/1850.png` |
| Figure 3 | `src/figures.py` | `fig_3/Political Economy.png`<br>`Religion.png`<br>`Science.png` |
| Figure 4 | `Rscripts/famous_books.R` | `fig_4/famous_volumes_gray.png` |
| Figure 5 | `src/figures.py` | `fig_5/avg_progress_raw.png` |
| Figure 6 | `src/figures.py` | `fig_6/1550.png`<br>`fig_6/1600.png`<br>`fig_6/1650.png`<br>`fig_6/1700.png`<br>`fig_6/1750.png`<br>`fig_6/1800.png`<br>`fig_6/1850.png` |
| Figure 7 | `Rscripts/marginal_predicted_figs.R` | `fig_7/predicted_values.png` |
| Figure 8 | `Rscripts/marginal_predicted_figs.R` | `fig_8/economics/predicted_values.png`<br>`fig_8/law/predicted_values.png`<br>`fig_8/literature/predicted_values.png` |
| Figure 9 | `src/figures.py` | `fig_9/1550.png`<br>`fig_9/1600.png`<br>`fig_9/1650.png`<br>`fig_9/1700.png`<br>`fig_9/1750.png`<br>`fig_9/1800.png`<br>`fig_9/1850.png` |
| Figure 10 | `Rscripts/additional_ternary_figs.R` | `fig_10/1550.png`<br>`fig_10/1600.png`<br>`fig_10/1650.png`<br>`fig_10/1700.png`<br>`fig_10/1750.png`<br>`fig_10/1800.png`<br>`fig_10/1850.png` |
| Figure 11 | `Rscripts/marginal_predicted_figs.R` | `fig_11/predicted_values_sci_pe.png` |
| Table B.1 | `Rscripts/regression_tables.R` | `table_b_1/results.tex` |
| Table B.2 | `Rscripts/regression_tables.R` | `table_b_2/results_author_fe.tex` |
| Table B.4 | `Rscripts/regression_tables.R` | `table_b_4/results.tex` |
| Figure B.1 | `src/figures.py` | `fig_b_1/total_volumes.png` |
| Figure B.4 | `src/figures.py` | `fig_b_4/corpus_vs_transl_categories.py` |
| Figure B.5 | `src/figures.py` | `fig_b_5/1550.png`<br>`fig_b_5/1600.png`<br>`fig_b_5/1650.png`<br>`fig_b_5/1700.png`<br>`fig_b_5/1750.png`<br>`fig_b_5/1800.png`<br>`fig_b_5/1850.png` |
| Figure B.6 | `Rscripts/famous_books.R` | `fig_b_6/famous_volume.png` |
| Figure B.7 | `src/figures.py` | `fig_b_7/avg_progress.png` |
| Figure B.8 | `src/figures.py` | `fig_b_8/1550.png`<br>`fig_b_8/1600.png`<br>`fig_b_8/1650.png`<br>`fig_b_8/1700.png`<br>`fig_b_8/1750.png`<br>`fig_b_8/1800.png`<br>`fig_b_8/1850.png` |
| Figure B.9 | `Rscripts/marginal_predicted_figs.R` | `fig_b_9/predicted_values.png` |
| Figure B.10 | `src/figures.py` | `fig_b_10/1550.png`<br>`fig_b_10/1600.png`<br>`fig_b_10/1650.png`<br>`fig_b_10/1700.png`<br>`fig_b_10/1750.png`<br>`fig_b_10/1800.png`<br>`fig_b_10/1850.png` |
| Figure B.11 | `Rscripts/marginal_predicted_figs.R` | `fig_b_11/predicted_values.png` |
| Figure B.12 | `src/figures.py` | `fig_b_12/1550.png`<br>`fig_b_12/1600.png`<br>`fig_b_12/1650.png`<br>`fig_b_12/1700.png`<br>`fig_b_12/1750.png`<br>`fig_b_12/1800.png`<br>`fig_b_12/1850.png` |
| Figure B.13 | `Rscripts/marginal_predicted_figs.R` | `fig_b_13/predicted_values.png` |
| Figure B.14 | `src/figures.py` | `fig_b_14/1550.png`<br>`fig_b_14/1600.png`<br>`fig_b_14/1650.png`<br>`fig_b_14/1700.png`<br>`fig_b_14/1750.png`<br>`fig_b_14/1800.png`<br>`fig_b_14/1850.png` |
| Figure B.16 | `src/figures.py` | `fig_b_16/avg_progress_translations_raw.png` |
| Figure B.17 | `src/figures.py` | `fig_b_17/1550.png`<br>`fig_b_17/1600.png`<br>`fig_b_17/1650.png`<br>`fig_b_17/1700.png`<br>`fig_b_17/1750.png`<br>`fig_b_17/1800.png`<br>`fig_b_17/1850.png` |
| Figure B.18 | `Rscripts/additional_ternary_figs.R` | `fig_b_18/1550.png`<br>`fig_b_18/1600.png`<br>`fig_b_18/1650.png`<br>`fig_b_18/1700.png`<br>`fig_b_18/1750.png`<br>`fig_b_18/1800.png`<br>`fig_b_18/1850.png` |
| Figure B.19 | `Rscripts/marginal_predicted_figs.R` | `fig_b_19/predicted_values.png` |
| Figure B.20 | `Rscripts/marginal_predicted_figs.R` | `fig_b_20/predicted_values_sci_pol.png`<br>`fig_b_20/predicted_values_sci_rel.png` |
| Figure B.21 | `src/figures.py` | `fig_b_21/1550.png`<br>`fig_b_21/1600.png`<br>`fig_b_21/1650.png`<br>`fig_b_21/1700.png`<br>`fig_b_21/1750.png`<br>`fig_b_21/1800.png`<br>`fig_b_21/1850.png` |
| Figure B.22 | `src/figures.py` | `fig_b_22/1550.png`<br>`fig_b_22/1600.png`<br>`fig_b_22/1650.png`<br>`fig_b_22/1700.png`<br>`fig_b_22/1750.png`<br>`fig_b_22/1800.png`<br>`fig_b_22/1850.png` |
| Figure B.23 | `src/figures.py` | `fig_b_23/1550.png`<br>`fig_b_23/1600.png`<br>`fig_b_23/1650.png`<br>`fig_b_23/1700.png`<br>`fig_b_23/1750.png`<br>`fig_b_23/1800.png`<br>`fig_b_23/1850.png` |
| Figure B.24 | `src/figures.py` | `fig_b_24/avg_industry_raw.png` |
| Figure B.25 | `src/figures.py` | `fig_b_25/1550.png`<br>`fig_b_25/1600.png`<br>`fig_b_25/1650.png`<br>`fig_b_25/1700.png`<br>`fig_b_25/1750.png`<br>`fig_b_25/1800.png`<br>`fig_b_25/1850.png` |
| Figure B.26 | `src/figures.py` | `fig_b_26/1550.png`<br>`fig_b_26/1600.png`<br>`fig_b_26/1650.png`<br>`fig_b_26/1700.png`<br>`fig_b_26/1750.png`<br>`fig_b_26/1800.png`<br>`fig_b_26/1850.png` |
| Figure B.27 | `Rscripts/marginal_predicted_figs.R` | `fig_b_27/predicted_values_sci_pe.png` |
| Figure C.1 | `src/figures.py` | `fig_c_1/1550.png`<br>`fig_c_1/1600.png`<br>`fig_c_1/1650.png`<br>`fig_c_1/1700.png`<br>`fig_c_1/1750.png`<br>`fig_c_1/1800.png`<br>`fig_c_1/1850.png` |
| Figure C.2 | `Rscripts/marginal_predicted_figs.R` | `fig_c_2/predicted_values.png` |
| Figure D.2 | `src/figures.py` | `fig_d_2/estc_hathitrust_pdf.png` |
| Figure E.1 | `src/figures.py` | `fig_e_1/1550.png`<br>`fig_e_1/1600.png`<br>`fig_e_1/1650.png`<br>`fig_e_1/1700.png`<br>`fig_e_1/1750.png`<br>`fig_e_1/1800.png`<br>`fig_e_1/1850.png` |
| Figure E.2 | `src/figures.py` | `fig_e_2/Political Economy.png`<br>`fig_e_2/Religion.png`<br>`fig_e_2/Science.png` |
| Figure E.3 | `src/figures.py` | `fig_e_3/1550.png`<br>`fig_e_3/1600.png`<br>`fig_e_3/1650.png`<br>`fig_e_3/1700.png`<br>`fig_e_3/1750.png`<br>`fig_e_3/1800.png`<br>`fig_e_3/1850.png` |
| Figure E.4 | `Rscripts/marginal_predicted_figs.R` | `fig_e_4/predicted_values.png` |
| Figure E.5 | `src/figures.py` | `fig_e_5/1550.png`<br>`fig_e_5/1600.png`<br>`fig_e_5/1650.png`<br>`fig_e_5/1700.png`<br>`fig_e_5/1750.png`<br>`fig_e_5/1800.png`<br>`fig_e_5/1850.png` |
| Figure E.6 | `Rscripts/additional_ternary_figs.R` | `fig_e_6/1550.png`<br>`fig_e_6/1600.png`<br>`fig_e_6/1650.png`<br>`fig_e_6/1700.png`<br>`fig_e_6/1750.png`<br>`fig_e_6/1800.png`<br>`fig_e_6/1850.png` |
| Figure E.7 | `Rscripts/marginal_predicted_figs.R` | `fig_e_7/predicted_values_sci_pe.png` |
| Figure F.1 | `src/figures.py` | `fig_f_1/1550.png`<br>`fig_f_1/1600.png`<br>`fig_f_1/1650.png`<br>`fig_f_1/1700.png`<br>`fig_f_1/1750.png`<br>`fig_f_1/1800.png`<br>`fig_f_1/1850.png` |
| Figure F.2 | `src/figures.py` | `fig_f_2/1550.png`<br>`fig_f_2/1600.png`<br>`fig_f_2/1650.png`<br>`fig_f_2/1700.png`<br>`fig_f_2/1750.png`<br>`fig_f_2/1800.png`<br>`fig_f_2/1850.png` |
| Figure G.1 | `src/figures.py` | `fig_g_1/1550.png`<br>`fig_g_1/1600.png`<br>`fig_g_1/1650.png`<br>`fig_g_1/1700.png`<br>`fig_g_1/1750.png`<br>`fig_g_1/1800.png`<br>`fig_g_1/1850.png` |
| Figure G.2 | `Rscripts/additional_ternary_figs.R` | `fig_g_2/1550.png`<br>`fig_g_2/1600.png`<br>`fig_g_2/1650.png`<br>`fig_g_2/1700.png`<br>`fig_g_2/1750.png`<br>`fig_g_2/1800.png`<br>`fig_g_2/1850.png` |
| Figure G.3 | `Rscripts/marginal_predicted_figs.R` | `fig_g_3/predicted_values_sci_pe.png` |
| Figure H.1 | `src/figures.py` | `fig_h_1/corpus_vs_manual_categories.png` |
| Figure H.2 | `src/figures.py` | `fig_h_2/1550.png`<br>`fig_h_2/1600.png`<br>`fig_h_2/1650.png`<br>`fig_h_2/1700.png`<br>`fig_h_2/1750.png`<br>`fig_h_2/1800.png`<br>`fig_h_2/1850.png` |
| Figure H.3 | `Rscripts/marginal_predicted_figs.R` | `fig_h_3/predicted_values.png` |


