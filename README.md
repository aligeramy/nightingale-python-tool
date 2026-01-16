# Nightingale Python Tool Service

A FastAPI service for sandboxed Python code execution and statistical analysis, designed to work with the Nightingale AI chatbot.

## Features

- **Sandboxed Python Execution**: Execute Python code safely with RestrictedPython
- **Statistical Analysis**: Built-in endpoints for t-tests, ANOVA, regression, descriptive statistics, and correlation
- **Security**: API key authentication and restricted code execution environment
- **Docker Support**: Ready for containerized deployment

## API Endpoints

### Health Check
```
GET /health
```

### Execute Python Code
```
POST /execute
Headers:
  X-API-Key: <your-api-key>
Body:
  {
    "code": "result = np.mean([1, 2, 3, 4, 5])",
    "timeout": 30
  }
```

### Statistical Analysis Endpoints

- `POST /analyze/ttest` - T-test analysis
- `POST /analyze/anova` - ANOVA analysis
- `POST /analyze/regression` - Regression analysis
- `POST /analyze/descriptive` - Descriptive statistics
- `POST /analyze/correlation` - Correlation analysis

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t python-tool .
docker run -p 8000:8000 -e API_KEY=your-key python-tool
```

## Deployment

This service is deployed on Railway. See `DEPLOY_RAILWAY.md` for deployment instructions.

## Environment Variables

- `api_key`: API key for authentication (required in production)
- `execution_timeout`: Code execution timeout in seconds (default: 30)
- `max_memory_mb`: Memory limit in MB (default: 512)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins

## Security

- Code execution is sandboxed using RestrictedPython
- Forbidden imports: os, sys, subprocess, socket, etc.
- Forbidden builtins: eval, exec, open, etc.
- Pre-loaded safe libraries: numpy, pandas, scipy.stats, math

## License

See LICENSE file for details.
