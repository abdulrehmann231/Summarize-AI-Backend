# Deployment Guide

## Backend (Render)

### Environment Variables

The following environment variables must be set in your Render Service Dashboard:

| Variable                   | Description                               | Example    |
| -------------------------- | ----------------------------------------- | ---------- |
| `HUGGINGFACEHUB_API_TOKEN` | Your Hugging Face API Token (Read access) | `hf_...`   |
| `PINECONE_API_KEY`         | Your Pinecone API Key                     | `pcsk_...` |

### Start Command

If not using `render.yaml`, ensure your Start Command is set to:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

> **Note:** The default `uvicorn main:app` only listens on `127.0.0.1`, which will cause 502 Bad Gateway errors on Render.

## Frontend (Vercel)

### Environment Variables

The following environment variables must be set in your Vercel Project Settings:

| Variable                  | Description                                          |
| ------------------------- | ---------------------------------------------------- | ------------------------------------ |
| `NEXT_PUBLIC_BACKEND_URL` | The URL of your deployed backend (no trailing slash) | `https://your-app-name.onrender.com` |

## Troubleshooting

### CORS Errors / 502 Bad Gateway

If you see a CORS error accompanied by a 502 Bad Gateway:

1. **It's likely not a CORS issue.** It's a startup issue.
2. The 502 means Render can't reach your app.
3. Because the app didn't reply, no CORS headers were sent.
4. **Fix:** Ensure the backend `Start Command` includes `--host 0.0.0.0`.
