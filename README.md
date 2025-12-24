# synthetic-data-evaluation
Information-theoretic evaluation of synthetic, simulated, and ground-truth data with extensions to fidelity.

## Directory Structure

### `synth_data_privacy_utility`

This directory contains separate pipelines for the evaluation of synthetic data privacy and utility for two datasets: the Adult dataset and the Bank dataset.

- **adult/**
  - **notebooks/**
    - `01_data_prep_adult.ipynb`: Data preparation for the Adult dataset.
    - `02_generate_synthetic_data_adult.ipynb`: Synthetic data generation for Adult.
    - `03_membership_inference_attack_adult.ipynb`: Membership inference attack analysis for Adult.
    - `04_utility_evaluation_adult.ipynb`: Utility evaluation for the Adult dataset.
    - `05_tradeoff_analysis_and_figures_adult.ipynb`: Trade-off analysis for Adult.
    - `06_result_analysis_adult.ipynb`: Result analysis for Adult.
  - **data_outputs/**: Generated outputs and results for the Adult dataset.

- **bank/**
  - **notebooks/**
    - `01b_bank_data_preparation.ipynb`: Data preparation for the Bank dataset.
    - `02b_generate_synthetic_data.ipynb`: Synthetic data generation for Bank.
    - `03b_membership_inference_attack_bank.ipynb`: Membership inference attack analysis for Bank.
    - `04b_utility_evaluation_bank.ipynb`: Utility evaluation for the Bank dataset.
    - `05b_tradeoff_analysis_bank.ipynb`: Trade-off analysis for Bank.
    - `bank_make_tables.ipynb`: Table creation for Bank results.
  - **data_outputs_bank/**: Generated outputs and results for the Bank dataset.

- **shared_resources/**
  - Utilities and shared functions used across both datasets. 

This structure allows for a coherent separation of tasks and results specific to each dataset, facilitating easier understanding and usage.