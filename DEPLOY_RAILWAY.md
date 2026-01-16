# Deploy Python Tool to Railway

## Quick Deploy Steps

### 1. Login to Railway
```bash
cd python_tool
railway login
```
This will open your browser for authentication.

### 2. Initialize Railway Project
```bash
railway init
```
- Choose "Create a new project"
- Name it something like `nightingale-python-tool`

### 3. Set Environment Variables

Generate a secure API key (already generated):
```
56a905fcf4e07e9d8456afba030f3a296d98207e20974d268633083f84fa7b15
```

Set the environment variables:
```bash
railway variables set api_key=56a905fcf4e07e9d8456afba030f3a296d98207e20974d268633083f84fa7b15
railway variables set execution_timeout=30
railway variables set max_memory_mb=512
railway variables set ALLOWED_ORIGINS=https://nightingale.softx.ca
```

### 4. Deploy
```bash
railway up
```

### 5. Get the Service URL
```bash
railway domain
```
This will generate a Railway domain (e.g., `your-app.up.railway.app`)

### 6. Update Vercel Environment Variables

After getting the Railway URL, add it to your Vercel project:

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add:
   - `PYTHON_SERVICE_URL=https://your-railway-url.up.railway.app`
   - `PYTHON_API_KEY=56a905fcf4e07e9d8456afba030f3a296d98207e20974d268633083f84fa7b15`
3. Redeploy your Vercel app

### 7. Test the Deployment

```bash
# Test health endpoint
curl https://your-railway-url.up.railway.app/health

# Should return:
# {"status":"healthy","service":"python-tool"}
```

## Troubleshooting

### Check Logs
```bash
railway logs
```

### View Service Info
```bash
railway status
```

### Update Environment Variables
```bash
railway variables
```
