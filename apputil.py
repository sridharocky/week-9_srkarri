import pandas as pd
import numpy as np


class GroupEstimate:
    def __init__(self, estimate="mean"):
        """
        Initialize the GroupEstimate model.

        Parameters:
            estimate (str): "mean" or "median" (default: "mean")
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.default_category = None
        self.default_estimates = None
        self._fit_columns = None

    def fit(self, X, y, default_category=None):
        """
        Fit the model using categorical data (X) and corresponding continuous values (y).

        Parameters:
            X (pd.DataFrame): Categorical features
            y (array-like): Continuous target variable
            default_category (str, optional): Column to use for fallback if group missing
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y must be of the same length")

        self.default_category = default_category
        self._fit_columns = list(X.columns)

        df = X.copy()
        df["_y_"] = y

        # Group by all categorical columns and compute estimate
        grouped = df.groupby(list(X.columns), observed=True)["_y_"]
        if self.estimate == "mean":
            group_values = grouped.mean()
        else:
            group_values = grouped.median()

        self.group_estimates = group_values.to_dict()

        # Compute per-category fallback if default_category specified
        if default_category is not None:
            if default_category not in X.columns:
                raise ValueError(f"{default_category} not found in X columns")

            grouped_def = df.groupby(default_category, observed=True)["_y_"]
            if self.estimate == "mean":
                default_values = grouped_def.mean()
            else:
                default_values = grouped_def.median()

            self.default_estimates = default_values.to_dict()

    def predict(self, X_):
        """
        Predict the estimated y-values for new observations.

        Parameters:
            X_ (array-like or pd.DataFrame): New categorical observations

        Returns:
            np.ndarray: Estimated y-values (NaN for missing combinations)
        """
        if self.group_estimates is None:
            raise RuntimeError("Model not fitted yet. Call .fit() before .predict().")

        # ✅ Ensure correct DataFrame with original columns
        if isinstance(X_, pd.DataFrame):
            X_new = X_.copy()
        else:
            X_new = pd.DataFrame(X_, columns=self._fit_columns)

        preds = []
        missing_count = 0

        for _, row in X_new.iterrows():
            key = tuple(row[col] for col in self._fit_columns)
            if key in self.group_estimates:
                preds.append(self.group_estimates[key])
            elif self.default_category and row[self.default_category] in self.default_estimates:
                preds.append(self.default_estimates[row[self.default_category]])
            else:
                preds.append(np.nan)
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s).")

        return np.array(preds)
