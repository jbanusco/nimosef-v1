
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as smt


def pairwise_comparisons(df, value_col, group_col='Segmentation', subject_col='Subject'):
    results = []
    # Loop over each class (e.g., LV, Myo, RV)
    for cl in df['Class'].unique():
        df_cl = df[df['Class'] == cl]
        # Pivot: rows are subjects, columns are segmentation groups
        pivot_df = df_cl.pivot(index=subject_col, columns=group_col, values=value_col)
        pivot_df = pivot_df.dropna()  # Only include subjects with complete data
        groups = pivot_df.columns.tolist()
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1, g2 = groups[i], groups[j]
                # Paired test: Wilcoxon signed-rank test for paired samples
                stat, p = stats.wilcoxon(pivot_df[g1], pivot_df[g2])

                # As an approximation, if you can compute a Z value:
                # Z = stat (if provided by your implementation) or compute it based on the test statistic.
                # Here we use a placeholder value for Z; in practice, you'll need the proper formula.
                N = pivot_df.shape[0]
                Z = stat / (N ** 0.5)  # This is a rough approximation
                r = Z / (N ** 0.5)   # Effect size r

                results.append({
                    'Class': cl,
                    'Measure': value_col,
                    'Comparison': f'{g1} vs {g2}',
                    'Statistic': stat,
                    'p-value': p,
                    'Effect Size r': r
                })
    results_df = pd.DataFrame(results)
    
    # Correct for multiple comparisons across all tests using Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = smt.multipletests(results_df['p-value'], method='fdr_bh')
    results_df['adj_p-value'] = pvals_corrected
    results_df['significant'] = reject
    
    return results_df


# Define a function to bootstrap the Spearman correlation confidence interval.
def bootstrap_spearman(x, y, n_bootstrap=1000, alpha=0.05):
    boot_corrs = []
    n = len(x)
    for _ in range(n_bootstrap):
        # Sample indices with replacement.
        indices = np.random.choice(n, n, replace=True)
        boot_corr, _ = stats.spearmanr(x[indices], y[indices])
        boot_corrs.append(boot_corr)
    lower = np.percentile(boot_corrs, 100 * (alpha / 2))
    upper = np.percentile(boot_corrs, 100 * (1 - alpha / 2))
    return lower, upper


def drop_high_nan(df, threshold=0.9):
    """
    Drops columns from df that have a fraction of NaN values above the given threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Fraction threshold (default is 0.9, i.e. drop columns with >90% NaNs).
    
    Returns:
        pd.DataFrame: DataFrame with high-NaN columns removed.
    """
    # Compute fraction of NaN per column and drop those above threshold.
    nan_fraction = df.isna().mean()
    cols_to_keep = nan_fraction[nan_fraction < threshold].index
    return df.loc[:, cols_to_keep]


def process_columns_for_rf(df, columns_to_drop, columns_to_code, nan_threshold=0.9):
    """
    Processes a DataFrame for RandomForest modeling.
    1. Drops columns specified in columns_to_drop.
    2. Drops any columns with > nan_threshold fraction of NaN values.
    3. For each column in columns_to_code:
         - Tries to convert to numeric.
         - If conversion fails, converts the column to categorical and replaces with its codes.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_drop (list): List of column names to drop.
        columns_to_code (list): List of column names that require processing:
                                 try numeric conversion, else convert to categorical codes.
        nan_threshold (float): Fraction threshold to drop columns with too many NaNs (default=0.9).
    
    Returns:
        pd.DataFrame: Processed DataFrame with only numeric columns.
    """
    # Drop specified columns.
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Drop columns with too many NaN values.
    df = drop_high_nan(df, threshold=nan_threshold)
    
    # for col in columns_to_code:
    for col in df.columns:
        if col in df.columns:
            try:
                # Try converting to numeric (float).
                df[col] = pd.to_numeric(df[col])
            except Exception:
                # If conversion fails, convert to categorical and use its codes.
                if col != 'Subject':
                    df[col] = pd.Categorical(df[col]).codes.astype(float)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    
    return df