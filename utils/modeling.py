import pandas as pd
from typing import Optional, Union, List

class GroupEstimate:
    def __init__(self, estimate: str = 'mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates: Optional[pd.Series] = None
        self.default_category: Optional[str] = None
        self.default_estimates: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Union[List, pd.Series], default_category: Optional[str] = None):
        """
        Fit the GroupEstimate model.
        Parameters:
        - X: pd.DataFrame of categorical features
        - y: 1D array or pd.Series of continuous target values
        - default_category: Optional column name for fallback estimates
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        df = X.copy()
        df['_y'] = y
        self.default_category = default_category

        # Group by all columns and compute estimate
        grouped = df.groupby(list(X.columns), observed=True)['_y']
        self.group_estimates = grouped.mean() if self.estimate == 'mean' else grouped.median()

        # If a default category is provided, compute fallback estimates
        if default_category and default_category in X.columns:
            default_grouped = df.groupby(default_category, observed=True)['_y']
            self.default_estimates = default_grouped.mean() if self.estimate == 'mean' else default_grouped.median()

    def predict(self, X_: Union[List[List], pd.DataFrame]) -> List[float]:
        """
        Predict estimates for new observations.
        Parameters:
        - X_: List of lists or DataFrame of categorical features
        Returns:
        - List of estimated y values
        """
        # Ensure X_ is a DataFrame with correct columns
        if isinstance(X_, list):
            df_ = pd.DataFrame(X_, columns=self.group_estimates.index.names)
        else:
            df_ = X_.copy()
            df_.columns = self.group_estimates.index.names

        results = []
        missing_count = 0

        for _, row in df_.iterrows():
            key = tuple(row)
            # Exact match for group
            if key in self.group_estimates:
                results.append(self.group_estimates[key])
            # Fallback to default category if defined
            elif self.default_category and row[self.default_category] in self.default_estimates:
                results.append(self.default_estimates[row[self.default_category]])
            else:
                results.append(float('nan'))
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s).")

        return results
