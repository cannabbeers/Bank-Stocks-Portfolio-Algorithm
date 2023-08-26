import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)

# Hardcoded base directory for the models
base_dir = "/mount/src/bank-stock-machine-learning-algo-and-stock-price-trend-predictor/resources/Models"

# Filenames for each dictionary
filenames = [
    'linear_regression.pkl',
    'lgbm_regressor.pkl',
    'svr.pkl',
    'nu_svr.pkl',
    'linear_svr.pkl',
    'sgd_regressor.pkl',
    'decision_tree_regressor.pkl',
    'gradient_boosting_regressor.pkl',
    'mlp_regressor.pkl',
    'kernel_ridge.pkl',
    'bayesian_ridge.pkl'
]

# Variable names for each dictionary
variables = [
    'linear_regression_dict',
    'lgbm_regressor_dict',
    'svr_dict',
    'nu_svr_dict',
    'linear_svr_dict',
    'sgd_regressor_dict',
    'decision_tree_regressor_dict',
    'gradient_boosting_regressor_dict',
    'mlp_regressor_dict',
    'kernel_ridge_dict',
    'bayesian_ridge_dict'
]

# Load each dictionary from its file
for filename, var_name in zip(filenames, variables):
    file_path = os.path.join(base_dir, filename)
    logging.info(f"Trying to open: {file_path}")
    with open(file_path, 'rb') as file:
        globals()[var_name] = pickle.load(file)

# Now, each dictionary is loaded back into its respective variable
