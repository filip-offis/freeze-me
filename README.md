# Freeze Me

Freeze Me is a local frontend/backend app for turning a short video into motion-based image effects. The repository is trimmed for student development: run the Vue frontend locally, run the FastAPI backend locally, and use SAM2 only when you need segmentation.

## Stack

- Frontend: Vue 3 + Vite + Vuetify
- Backend: FastAPI
- Video/image processing: OpenCV, ffmpeg-python, Pillow
- Segmentation: SAM2 + PyTorch

## Prerequisites

- Node.js 20+
- Python 3.10+

You do not need Docker, Podman, nginx, or Conda for the default setup.

## Project layout

```text
freeze-me/
  backend/   FastAPI app and Python dependencies
  frontend/  Vue app
  notebook/  project notes and experiments
```

## Backend setup

From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you want GPU-only blur acceleration, install the optional extra after the base requirements:

```powershell
pip install -r requirements-gpu.txt
```

### SAM2 checkpoint

The backend starts without loading SAM2. The model is loaded only when a segmentation endpoint is used.

Before using segmentation, download a checkpoint into `backend/checkpoints`:

```powershell
python download_checkpoint.py
```

Optional: choose a different checkpoint size.

```powershell
$env:SAM_VERSION="tiny"
python download_checkpoint.py tiny
```

Supported values are `tiny`, `small`, `b_plus`, and `large`. If `SAM_VERSION` is not set, `small` is used.

If you choose anything other than `small`, keep the same `SAM_VERSION` set when starting the backend:

```powershell
$env:SAM_VERSION="tiny"
python -m uvicorn src.main:app --reload
```

### Start the backend

Run this from `backend/`:

```powershell
python -m uvicorn src.main:app --reload
```

The backend runs at `http://localhost:8000`.

## Frontend setup

Open a second terminal from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`.

By default, the frontend calls `http://localhost:8000`.

If your backend runs somewhere else, create `frontend/.env.local`:

```env
VITE_API_URL=http://localhost:8000
```

## Local development flow

1. Start the backend in `backend/`.
2. Start the frontend in `frontend/`.
3. Open `http://localhost:5173`.
4. Upload a video and work through the UI.

## Notes

- Uploaded files and generated outputs are stored in `backend/videos/`.
- The app can still start without a SAM2 checkpoint, but segmentation will fail until the checkpoint is downloaded.
- CPU-only setups are fine for local development, but segmentation and effect generation will be slower.
- The `notebook/` directory is kept for documentation and experimentation, but it is not required to run the app.

## Troubleshooting

- `SAM2 checkpoint not found`: run `python download_checkpoint.py` inside `backend/`.
- Frontend cannot reach backend: make sure the backend is running on port `8000`, or set `VITE_API_URL` in `frontend/.env.local`.
- Empty gallery on a fresh setup: this is expected until you upload your first video.
