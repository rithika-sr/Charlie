#!/bin/bash
# Deploy MBTA Dashboard with Google Maps Trip Planner to Cloud Run

PROJECT_ID="charlie-478223"
IMAGE_NAME="gcr.io/${PROJECT_ID}/charlie-mbta-chatbot:latest"
REGION="us-east1"

echo "🚀 Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE_NAME .

echo "📤 Pushing to Google Container Registry..."
docker push $IMAGE_NAME

echo "☁️ Deploying to Cloud Run..."
gcloud run deploy charlie-mbta-chatbot \
    --image $IMAGE_NAME \
    --region $REGION \
    --port 8080 \
    --memory 1Gi \
    --allow-unauthenticated \
    --set-env-vars "MBTA_API_KEY=${MBTA_API_KEY},OPENAI_API_KEY=${OPENAI_API_KEY},GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}"

echo "✅ Deployment complete!"
