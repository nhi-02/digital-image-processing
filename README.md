# Whisper of Pixel — Digital Image Processing (FastAPI + React + TypeScript)

Professional small project for educational and demo purposes: a FastAPI backend providing classic image-processing operations and a React + Vite frontend UI for uploading images, running operations and saving results.

---

## Highlights
- Backend (FastAPI) exposes a rich set of image-processing endpoints (color, point processing, histogram, noise, spatial/frequency filters, morphology, segmentation, PCA, YOLOv8 detection, Huffman compression).
  - Entrypoints / routing and request handling: [`upload_image`](backend/main.py) and many route handlers in [`backend/main.py`](backend/main.py).
  - Core ops implemented in: [`image_ops.py`](backend/image_ops.py) — e.g. [`rgb_channels`](backend/image_ops.py), [`gaussian_filter`](backend/image_ops.py), [`fft2_magnitude_phase`](backend/image_ops.py), [`detect_person`](backend/image_ops.py), [`huffman_encode`](backend/image_ops.py).
  - Runtime image storage: [`ImageStore`](backend/image_store.py).
- Frontend (React + TypeScript + Vite) provides an interactive UI:
  - Main app: [`whisper-of-pixel/src/App.tsx`](whisper-of-pixel/src/App.tsx)
  - API client: [`whisper-of-pixel/src/api/client.ts`](whisper-of-pixel/src/api/client.ts) — functions such as [`uploadImage`](whisper-of-pixel/src/api/client.ts).
  - Operation metadata used to render the sidebar: [`PARENTS`](whisper-of-pixel/src/constants/operations.ts).

---

## Repo layout (important files)
- Backend
  - [backend/main.py](backend/main.py) — API routes + helper logic (upload, color, point, histogram, noise, filters, segmentation, PCA, YOLOv8, compression)
  - [backend/image_ops.py](backend/image_ops.py) — image algorithms & utilities
  - [backend/image_store.py](backend/image_store.py) — in-memory image store class [`ImageStore`](backend/image_store.py)
  - [backend/requirements.txt](backend/requirements.txt) — Python dependencies
- Frontend
  - [whisper-of-pixel/src/App.tsx](whisper-of-pixel/src/App.tsx) — main UI
  - [whisper-of-pixel/src/api/client.ts](whisper-of-pixel/src/api/client.ts) — HTTP helpers and API base
  - [whisper-of-pixel/src/constants/operations.ts](whisper-of-pixel/src/constants/operations.ts) — UI operation metadata
  - [whisper-of-pixel/package.json](whisper-of-pixel/package.json) — frontend scripts & deps
  - [whisper-of-pixel/README.md](whisper-of-pixel/README.md) — Vite/TS template docs

---

## Quickstart — Development

1. Backend
   - Create a virtual env and install deps:
     - python -m venv .venv && source .venv/bin/activate
     - pip install -r backend/requirements.txt
   - Run dev server (FastAPI + Uvicorn):
     - uvicorn backend.main:app --reload --host 0.0.0.0 --port 8500

2. Frontend
   - Open new terminal:
     - cd whisper-of-pixel
     - npm install
     - npm run dev
   - Open the Vite dev URL from terminal (default: http://localhost:5173) and ensure the frontend uses the backend: `VITE_API_BASE_URL` env var or default `http://localhost:8500`.

---

## Example usage
- Upload an image via `/upload` (see [`upload_image`](backend/main.py)).
- Run a grayscale conversion via GET `/color/gray` (route implemented in [`backend/main.py`](backend/main.py) calling [`to_gray`](backend/image_ops.py)).
- Run Gaussian filter via `/spatial/gaussian` which invokes [`gaussian_filter`](backend/image_ops.py) from the same file.
- Use the frontend UI to explore every operation described in [`PARENTS`](whisper-of-pixel/src/constants/operations.ts).

---

## Notes & limitations
- Image storage is in-memory with FIFO eviction — [`ImageStore`](backend/image_store.py).
- YOLOv8 detection uses local model file `yolov8n.pt` (large model artifact kept in repo for quick demo).
- Huffman compression helpers live in [`backend/image_ops.py`](backend/image_ops.py) and produce base64 compressed payload.

---

## Contributing
- Add tests, fix bugs, or extend algorithms in:
  - image ops: [`backend/image_ops.py`](backend/image_ops.py)
  - add new frontend operation entries to: [`whisper-of-pixel/src/constants/operations.ts`](whisper-of-pixel/src/constants/operations.ts)
  - React UI: [`whisper-of-pixel/src/App.tsx`](whisper-of-pixel/src/App.tsx)
- Keep API backward-compatible for the frontend.
