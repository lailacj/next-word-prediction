# Better Language Models Better Model the N400, but not Reading Time

We provide the code and data for the paper 'Better Language Models Better Model the N400, but not Reading Time.

The contents of this repository are as follows:

* `datasets` contains the preprocessed reading time and N400 data used in this study. Note that here, as in all other directories, `szewczyk_2022.tsv` includes all 5 datasets provided by Szewczyk and Federmeier (2022). 
* `stimuli` contains the experimental stimili in the format used to calculate surprisal using the language models.
* `stimuli_details` contains dataframes that contain the information needed to merge the stimulus, language model output, and N400/reading time files.
* `code` contains all the code needed to calculate language model surprisals on the experimental stimuli (`calculate_surprisal.py`), as well as benchmark performance (`eval_models.sh`). To calculate all surprisals (in parallel using `slurm`), run `run_all.sh`. To calculate performance at all benchmarks, run `run_all_evals.sh`.
* `resuls_merged` contains the surprisals from all language models, with different files for each dataset. Files are merged such that each surprisal for each model is provided in a separate column for all stimuli.
* `calculate_fits` includes R scripts for calculating fit to the N400 and reading time data. Each file beginning in `calculate_aics_` calculates the fit to the data with the linear mixed-effects models used in Experiments 1-5. The R scripts for some datasets are separated to allow multiple regressions to be fit in parallel to speed up analysis. `calculate_lm_fits_and_corrs.R` includes the R script for calculating the correlations and fits used in Experiment 6.
* `analyze_fits` contains `main_results.Rmd`, which runs all regressions over fits and generates all plots. It is fully rendered as `main_results.html`. The remainder of the directory contains output files from `main_results.Rmd`, including all figures and the `regression_patterns.tsv`, which provides the outputs of all statistical tests carried out (note that `regression_patterns_cleaned.tsv` contains the same data but a reduced number of columns and rounded numbers ).