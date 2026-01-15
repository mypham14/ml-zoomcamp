# Deployment instructions

## Streamlit Community Cloud (recommended)
1. Push your repository to GitHub (public or private).
2. Go to https://share.streamlit.io and click **'New app'**.
3. Connect your GitHub account and select this repository and the branch (e.g., `main`).
4. Set the main file path to `app/streamlit_app.py`.
5. Streamlit will use `requirements.txt` to install dependencies automatically.

## Docker (alternative)
Build the image:

    docker build -t cyber-threat-app .

Run the container:

    docker run -p 8501:8501 cyber-threat-app

Open http://localhost:8501 to view the app.

## Notes
- Ensure your `models/` folder is committed or available (or modify the app to download model artifacts at startup). If models are large, consider hosting them externally and downloading at runtime, or mounting at container runtime.

If you deploy to Streamlit Community Cloud, it will install the packages listed in `requirements.txt` automatically; ensure required packages are present there.

---

## Docker (local test)
Build the Docker image locally and run it to validate the containerized app:

1. Build the image:

    docker build -t cyber-threat-app .

2. Run the image (bind host port 8501 to container 8501):

    docker run --rm -p 8501:8501 cyber-threat-app

If you prefer to mount local `models/` at runtime instead of baking them into the image:

    docker run --rm -p 8501:8501 -v "${PWD}:/app/models":/app/models cyber-threat-app

Open http://localhost:8501 to view the app.

Notes:
- If the port is already used, change the host side of the port mapping (e.g., `-p 8502:8501`).
- Add a `.dockerignore` file to keep the image small (exclude `.venv`, `data/`, `output/`, etc.).
