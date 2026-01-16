"""Statistical analysis functions."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from typing import Any


def calculate_ttest(
    group1: list[float],
    group2: list[float],
    paired: bool = False,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Perform t-test between two groups.

    Args:
        group1: First group of values
        group2: Second group of values
        paired: Whether to perform paired t-test
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results
    """
    try:
        arr1 = np.array(group1)
        arr2 = np.array(group2)

        if paired:
            if len(arr1) != len(arr2):
                return {
                    "success": False,
                    "error": "Paired t-test requires equal-sized groups",
                }
            result = stats.ttest_rel(arr1, arr2, alternative=alternative)
            df = len(arr1) - 1
        else:
            result = stats.ttest_ind(arr1, arr2, alternative=alternative)
            # Welch's degrees of freedom approximation
            n1, n2 = len(arr1), len(arr2)
            v1, v2 = arr1.var(ddof=1), arr2.var(ddof=1)
            df = ((v1 / n1 + v2 / n2) ** 2) / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )

        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(
            ((len(arr1) - 1) * arr1.std(ddof=1) ** 2 + (len(arr2) - 1) * arr2.std(ddof=1) ** 2)
            / (len(arr1) + len(arr2) - 2)
        )
        cohens_d = (arr1.mean() - arr2.mean()) / pooled_std if pooled_std > 0 else 0

        return {
            "success": True,
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "degrees_of_freedom": float(df),
            "mean1": float(arr1.mean()),
            "mean2": float(arr2.mean()),
            "std1": float(arr1.std(ddof=1)),
            "std2": float(arr2.std(ddof=1)),
            "effect_size": float(cohens_d),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_anova(groups: list[list[float]]) -> dict[str, Any]:
    """
    Perform one-way ANOVA.

    Args:
        groups: List of groups, each containing numeric values

    Returns:
        Dictionary with ANOVA results
    """
    try:
        if len(groups) < 2:
            return {"success": False, "error": "ANOVA requires at least 2 groups"}

        arrays = [np.array(g) for g in groups]
        result = stats.f_oneway(*arrays)

        # Calculate eta-squared (effect size)
        all_data = np.concatenate(arrays)
        grand_mean = all_data.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in arrays)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        df_between = len(groups) - 1
        df_within = sum(len(g) - 1 for g in arrays)

        return {
            "success": True,
            "f_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "df_between": df_between,
            "df_within": df_within,
            "eta_squared": float(eta_squared),
            "group_means": [float(g.mean()) for g in arrays],
            "group_stds": [float(g.std(ddof=1)) for g in arrays],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_regression(
    x: list[list[float]] | list[float],
    y: list[float],
    regression_type: str = "linear",
) -> dict[str, Any]:
    """
    Perform regression analysis.

    Args:
        x: Independent variable(s)
        y: Dependent variable
        regression_type: 'linear' or 'logistic'

    Returns:
        Dictionary with regression results
    """
    try:
        # Convert to numpy arrays
        y_arr = np.array(y)

        # Handle both simple and multiple regression
        if isinstance(x[0], (list, tuple)):
            X = np.array(x)
        else:
            X = np.array(x).reshape(-1, 1)

        if regression_type == "logistic":
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y_arr)

            predictions = model.predict_proba(X)[:, 1].tolist()

            return {
                "success": True,
                "coefficients": model.coef_.flatten().tolist(),
                "intercept": float(model.intercept_[0]),
                "predictions": predictions,
                "r_squared": None,  # Not applicable for logistic
                "adj_r_squared": None,
                "p_values": None,  # Would need statsmodels for this
                "std_errors": None,
            }
        else:
            # Use statsmodels for full linear regression stats
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y_arr, X_with_const).fit()

            return {
                "success": True,
                "coefficients": model.params[1:].tolist(),
                "intercept": float(model.params[0]),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "p_values": model.pvalues[1:].tolist(),
                "std_errors": model.bse[1:].tolist(),
                "predictions": model.predict(X_with_const).tolist(),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_descriptive(data: list[float]) -> dict[str, Any]:
    """
    Calculate descriptive statistics.

    Args:
        data: Numeric data

    Returns:
        Dictionary with descriptive statistics
    """
    try:
        arr = np.array(data)

        # Handle mode (may have multiple modes)
        mode_result = stats.mode(arr, keepdims=True)
        mode_val = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))

        return {
            "success": True,
            "count": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "mode": mode_val,
            "std": float(arr.std(ddof=1)),
            "variance": float(arr.var(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_correlation(
    x: list[float],
    y: list[float],
    method: str = "pearson",
) -> dict[str, Any]:
    """
    Calculate correlation between two variables.

    Args:
        x: First variable values
        y: Second variable values
        method: 'pearson', 'spearman', or 'kendall'

    Returns:
        Dictionary with correlation results
    """
    try:
        arr_x = np.array(x)
        arr_y = np.array(y)

        if len(arr_x) != len(arr_y):
            return {"success": False, "error": "Arrays must have the same length"}

        if method == "pearson":
            corr, p_val = stats.pearsonr(arr_x, arr_y)
        elif method == "spearman":
            corr, p_val = stats.spearmanr(arr_x, arr_y)
        elif method == "kendall":
            corr, p_val = stats.kendalltau(arr_x, arr_y)
        else:
            return {"success": False, "error": f"Unknown method: {method}"}

        return {
            "success": True,
            "correlation": float(corr),
            "p_value": float(p_val),
            "method": method,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
