"""Launch script for the DJ-AI web application."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("dj_ai.app:app", host="127.0.0.1", port=8000, reload=False)
