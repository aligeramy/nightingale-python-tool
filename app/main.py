"""FastAPI service for Python code execution and statistical analysis."""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
import os
from app.models.schemas import (
    ExecuteRequest,
    ExecuteResponse,
    TTestRequest,
    TTestResponse,
    AnovaRequest,
    AnovaResponse,
    RegressionRequest,
    RegressionResponse,
    DescriptiveRequest,
    DescriptiveResponse,
    CorrelationRequest,
    CorrelationResponse,
)
from app.services.executor import execute_code
from app.services.statistics import (
    calculate_ttest,
    calculate_anova,
    calculate_regression,
    calculate_descriptive,
    calculate_correlation,
)


app = FastAPI(
    title="Python Tool Service",
    description="Sandboxed Python execution and statistical analysis API",
    version="1.0.0",
)

# CORS middleware
# Allow origins from environment variable or default to allow all in dev
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify the API key for protected endpoints."""
    if settings.api_key != "dev-key-change-me":  # Only enforce in production
        if x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "python-tool"}


@app.post("/execute", response_model=ExecuteResponse)
async def execute_python(
    request: ExecuteRequest,
    _: str = Depends(verify_api_key),
):
    """
    Execute arbitrary Python code in a sandboxed environment.

    The code runs with limited capabilities:
    - No file system access
    - No network access
    - No dangerous builtins (eval, exec, etc.)
    - Available libraries: numpy, pandas, scipy.stats, math

    To return a value, assign it to a variable named 'result'.
    """
    result = execute_code(request.code, request.timeout)

    return ExecuteResponse(
        success=result["success"],
        output=result.get("output"),
        result=result.get("result"),
        error=result.get("error"),
        message="Code executed successfully" if result["success"] else None,
    )


@app.post("/analyze/ttest", response_model=TTestResponse)
async def analyze_ttest(
    request: TTestRequest,
    _: str = Depends(verify_api_key),
):
    """
    Perform a t-test between two groups.

    Supports:
    - Independent samples t-test (default)
    - Paired samples t-test (paired=True)
    - One-tailed tests (alternative='less' or 'greater')

    Returns t-statistic, p-value, degrees of freedom, and Cohen's d effect size.
    """
    result = calculate_ttest(
        request.group1,
        request.group2,
        request.paired,
        request.alternative,
    )

    if not result["success"]:
        return TTestResponse(success=False, error=result.get("error"))

    return TTestResponse(
        success=True,
        message="T-test completed successfully",
        t_statistic=result.get("t_statistic"),
        p_value=result.get("p_value"),
        degrees_of_freedom=result.get("degrees_of_freedom"),
        mean1=result.get("mean1"),
        mean2=result.get("mean2"),
        std1=result.get("std1"),
        std2=result.get("std2"),
        effect_size=result.get("effect_size"),
    )


@app.post("/analyze/anova", response_model=AnovaResponse)
async def analyze_anova(
    request: AnovaRequest,
    _: str = Depends(verify_api_key),
):
    """
    Perform one-way ANOVA.

    Tests whether the means of multiple groups are significantly different.
    Returns F-statistic, p-value, degrees of freedom, and eta-squared effect size.
    """
    result = calculate_anova(request.groups)

    if not result["success"]:
        return AnovaResponse(success=False, error=result.get("error"))

    return AnovaResponse(
        success=True,
        message="ANOVA completed successfully",
        f_statistic=result.get("f_statistic"),
        p_value=result.get("p_value"),
        df_between=result.get("df_between"),
        df_within=result.get("df_within"),
        eta_squared=result.get("eta_squared"),
        group_means=result.get("group_means"),
        group_stds=result.get("group_stds"),
    )


@app.post("/analyze/regression", response_model=RegressionResponse)
async def analyze_regression(
    request: RegressionRequest,
    _: str = Depends(verify_api_key),
):
    """
    Perform regression analysis.

    Supports:
    - Simple linear regression (single predictor)
    - Multiple linear regression (multiple predictors)
    - Logistic regression (for binary outcomes)

    Returns coefficients, R-squared, p-values, and predictions.
    """
    result = calculate_regression(
        request.x,
        request.y,
        request.regression_type,
    )

    if not result["success"]:
        return RegressionResponse(success=False, error=result.get("error"))

    return RegressionResponse(
        success=True,
        message=f"{request.regression_type.title()} regression completed successfully",
        coefficients=result.get("coefficients"),
        intercept=result.get("intercept"),
        r_squared=result.get("r_squared"),
        adj_r_squared=result.get("adj_r_squared"),
        p_values=result.get("p_values"),
        std_errors=result.get("std_errors"),
        predictions=result.get("predictions"),
    )


@app.post("/analyze/descriptive", response_model=DescriptiveResponse)
async def analyze_descriptive(
    request: DescriptiveRequest,
    _: str = Depends(verify_api_key),
):
    """
    Calculate descriptive statistics for a dataset.

    Returns count, mean, median, mode, standard deviation, variance,
    min, max, range, quartiles, IQR, skewness, and kurtosis.
    """
    result = calculate_descriptive(request.data)

    if not result["success"]:
        return DescriptiveResponse(success=False, error=result.get("error"))

    return DescriptiveResponse(
        success=True,
        message="Descriptive statistics calculated successfully",
        count=result.get("count"),
        mean=result.get("mean"),
        median=result.get("median"),
        mode=result.get("mode"),
        std=result.get("std"),
        variance=result.get("variance"),
        min=result.get("min"),
        max=result.get("max"),
        range=result.get("range"),
        q1=result.get("q1"),
        q3=result.get("q3"),
        iqr=result.get("iqr"),
        skewness=result.get("skewness"),
        kurtosis=result.get("kurtosis"),
    )


@app.post("/analyze/correlation", response_model=CorrelationResponse)
async def analyze_correlation(
    request: CorrelationRequest,
    _: str = Depends(verify_api_key),
):
    """
    Calculate correlation between two variables.

    Supports:
    - Pearson correlation (default, for linear relationships)
    - Spearman correlation (for monotonic relationships)
    - Kendall's tau (for ordinal data)

    Returns correlation coefficient and p-value.
    """
    result = calculate_correlation(
        request.x,
        request.y,
        request.method,
    )

    if not result["success"]:
        return CorrelationResponse(success=False, error=result.get("error"))

    return CorrelationResponse(
        success=True,
        message=f"{request.method.title()} correlation calculated successfully",
        correlation=result.get("correlation"),
        p_value=result.get("p_value"),
        method=result.get("method"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
