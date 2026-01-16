from typing import Any
from pydantic import BaseModel, Field


# Base response models
class BaseResponse(BaseModel):
    success: bool
    message: str | None = None
    error: str | None = None


# Code execution models
class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=30, ge=1, le=60, description="Execution timeout in seconds")


class ExecuteResponse(BaseResponse):
    output: str | None = None
    result: Any = None


# Statistical analysis models
class TTestRequest(BaseModel):
    group1: list[float] = Field(..., description="First group of numeric values")
    group2: list[float] = Field(..., description="Second group of numeric values")
    paired: bool = Field(default=False, description="Whether to perform paired t-test")
    alternative: str = Field(
        default="two-sided",
        description="Alternative hypothesis: 'two-sided', 'less', or 'greater'",
    )


class TTestResponse(BaseResponse):
    t_statistic: float | None = None
    p_value: float | None = None
    degrees_of_freedom: float | None = None
    mean1: float | None = None
    mean2: float | None = None
    std1: float | None = None
    std2: float | None = None
    effect_size: float | None = None  # Cohen's d


class AnovaRequest(BaseModel):
    groups: list[list[float]] = Field(
        ..., description="List of groups, each containing numeric values"
    )


class AnovaResponse(BaseResponse):
    f_statistic: float | None = None
    p_value: float | None = None
    df_between: int | None = None
    df_within: int | None = None
    eta_squared: float | None = None
    group_means: list[float] | None = None
    group_stds: list[float] | None = None


class RegressionRequest(BaseModel):
    x: list[list[float]] | list[float] = Field(
        ..., description="Independent variable(s). Single list for simple regression, list of lists for multiple."
    )
    y: list[float] = Field(..., description="Dependent variable values")
    regression_type: str = Field(
        default="linear", description="Type: 'linear' or 'logistic'"
    )


class RegressionResponse(BaseResponse):
    coefficients: list[float] | None = None
    intercept: float | None = None
    r_squared: float | None = None
    adj_r_squared: float | None = None
    p_values: list[float] | None = None
    std_errors: list[float] | None = None
    predictions: list[float] | None = None


class DescriptiveRequest(BaseModel):
    data: list[float] = Field(..., description="Numeric data for analysis")


class DescriptiveResponse(BaseResponse):
    count: int | None = None
    mean: float | None = None
    median: float | None = None
    mode: float | None = None
    std: float | None = None
    variance: float | None = None
    min: float | None = None
    max: float | None = None
    range: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


class CorrelationRequest(BaseModel):
    x: list[float] = Field(..., description="First variable values")
    y: list[float] = Field(..., description="Second variable values")
    method: str = Field(
        default="pearson", description="Correlation method: 'pearson', 'spearman', or 'kendall'"
    )


class CorrelationResponse(BaseResponse):
    correlation: float | None = None
    p_value: float | None = None
    method: str | None = None
