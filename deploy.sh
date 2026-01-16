#!/bin/bash

# Deploy Python Tool to Railway
# Make sure you're logged in: railway login

set -e

echo "🚀 Deploying Python Tool to Railway..."
echo ""

# Check if logged in
if ! railway whoami &>/dev/null; then
    echo "❌ Not logged in to Railway. Please run: railway login"
    exit 1
fi

echo "✅ Logged in to Railway"
echo ""

# Set environment variables
echo "📝 Setting environment variables..."
railway variables set api_key=56a905fcf4e07e9d8456afba030f3a296d98207e20974d268633083f84fa7b15
railway variables set execution_timeout=30
railway variables set max_memory_mb=512
railway variables set ALLOWED_ORIGINS=https://nightingale.softx.ca

echo "✅ Environment variables set"
echo ""

# Deploy
echo "🚀 Deploying..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get your Railway URL: railway domain"
echo "2. Add PYTHON_SERVICE_URL to Vercel environment variables"
echo "3. Add PYTHON_API_KEY to Vercel environment variables"
echo "4. Redeploy your Vercel app"
echo ""
echo "🧪 Test your deployment:"
echo "curl https://your-railway-url.up.railway.app/health"
