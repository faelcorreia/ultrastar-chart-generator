"""``python -m src.api``: serve the REST API."""
import os

import uvicorn

from src.api.server import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host=os.environ.get("API_HOST", "0.0.0.0"),
                port=int(os.environ.get("API_PORT", "8000")))
