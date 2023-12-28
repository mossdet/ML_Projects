import pandas as pd

from credit_g_dataset import get_preprocessed_credit_g_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 6)

X_train, X_validation, X_test, y_train, y_validation, y_test = get_preprocessed_credit_g_dataset()

pass
