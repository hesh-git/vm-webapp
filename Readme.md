gcloud builds submit --tag gcr.io/neurowhiz-422511/neurowhiz-app  --project=neurowhiz-422511


gcloud run deploy --image gcr.io/neurowhiz-422511/neurowhiz-app --platform managed  --project=neurowhiz-422511 --allow-unauthenticated