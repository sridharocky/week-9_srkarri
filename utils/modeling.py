import pandas as pd
import pickle

class GroupEstimate:
    def __init__(self, estimate='mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.default_category = None
        self.default_estimates = None

    def fit(self, X: pd.DataFrame, y, default_category=None):
        df = X.copy()
        df['_y'] = y
        self.default_category = default_category
        self.group_estimates = df.groupby(list(X.columns), observed=True)['_y']
        if self.estimate == 'mean':
            self.group_estimates = self.group_estimates.mean()
        else:
            self.group_estimates = self.group_estimates.median()

        if default_category:
            # Compute default category estimates
            self.default_estimates = df.groupby(default_category, observed=True)['_y']
            if self.estimate == 'mean':
                self.default_estimates = self.default_estimates.mean()
            else:
                self.default_estimates = self.default_estimates.median()

    def predict(self, X_):
        df_ = pd.DataFrame(X_, columns=self.group_estimates.index.names)
        results = []
        missing_count = 0

        for _, row in df_.iterrows():
            key = tuple(row)
            if key in self.group_estimates:
                results.append(self.group_estimates[key])
            elif self.default_category and row[self.default_category] in self.default_estimates:
                results.append(self.default_estimates[row[self.default_category]])
            else:
                results.append(float('nan'))
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s).")

        return results
