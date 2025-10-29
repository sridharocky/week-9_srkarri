import pandas as pd

class GroupEstimate:
    def __init__(self, estimate='mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.default_category = None
        self.default_estimates = None

    def fit(self, X: pd.DataFrame, y, default_category=None):
        """
        Fit the GroupEstimate model.

        Parameters:
        - X: pandas DataFrame of categorical columns
        - y: array-like of continuous values
        - default_category: optional column name to use for fallback estimates
        """
        df = X.copy()
        df['_y'] = y

        self.default_category = default_category

        # Compute main group estimates
        self.group_estimates = df.groupby(list(X.columns), observed=True)['_y']
        if self.estimate == 'mean':
            self.group_estimates = self.group_estimates.mean()
        else:
            self.group_estimates = self.group_estimates.median()

        # Compute default estimates if fallback category provided
        if default_category:
            self.default_estimates = df.groupby(default_category, observed=True)['_y']
            if self.estimate == 'mean':
                self.default_estimates = self.default_estimates.mean()
            else:
                self.default_estimates = self.default_estimates.median()

    def predict(self, X_):
        """
        Predict estimates for new observations.

        Parameters:
        - X_: array-like or DataFrame of new observations
        Returns:
        - List of predicted values (mean/median) or NaN if missing
        """
        df_ = pd.DataFrame(X_, columns=self.group_estimates.index.names)
        results = []
        missing_count = 0

        for _, row in df_.iterrows():
            key = tuple(row)
            # Case 1: Exact group match
            if key in self.group_estimates:
                results.append(self.group_estimates[key])
            # Case 2: Fallback using default_category
            elif self.default_category and row[self.default_category] in self.default_estimates:
                results.append(self.default_estimates[row[self.default_category]])
            # Case 3: Missing group
            else:
                results.append(float('nan'))
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s).")

        return results
