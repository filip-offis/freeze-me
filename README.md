# Freeze Me!
Freeze Me! is a project, in which we want to create dynamic, motion-illustrative images from video clips by isolating moving objects and visualizing their motion.


## Prerequisites

- **Node.js** (for the frontend) - [Download here](https://nodejs.org/en/download/)
- **Python 3.10+** (for the backend) - [Download here](https://www.python.org/downloads/)
- **Conda** (for environment management) - [Install here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Recommended, but optional:

- **CUDA 12.4** (for GPU Usage) - [Download here](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- **cudNN 9.7.0** (for GPU Usage) - [Install here](https://developer.nvidia.com/cudnn-downloads)


---

## Setup Instructions

### 1. Backend Setup

1. **Navigate to the backend folder:**
   ```sh
   cd backend
   ```

2. **Set up the Python environment:**
   If you're using Conda, create and activate the environment:
   ```sh
   conda create --name simple-webapp python=3.10
   conda activate simple-webapp
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Install Sam2**
   > Setup Sam2 and the correct pytorch-cuda version (if used with gpu)
   ```sh
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
   > Download a checkpoint and config from and add them to the config- and checkpoints
   > folder inside the backend folder.
   ```
   https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints
   ```
   
5. **Setup GPU Usage**
   > Ensure that both CUDA (12.4) and cuDNN (9.7.0) are installed and the PATH variables are set.
   
6. **Run the backend server:**
   ```sh
   cd src
   python main.py
   ```

   Your backend should now be running at `http://localhost:8000`.

---

### 2. Frontend Setup

1. **Navigate to the frontend folder:**
   ```sh
   cd frontend
   ```

2. **Install the dependencies:**
   ```sh
   npm install
   ```

3. **Start the development server:**
   ```sh
   npm run dev
   ```

   Your frontend should now be running at `http://localhost:[PORT]`.

---

### 3. Docker Compose (Alternative)

Instead of setting up the frontend and backend manually, you can run the entire application with Docker Compose:

1. **Prerequisites:**
   - [Docker](https://docs.docker.com/get-docker/) installed and running

2. **Start the application:**
   ```sh
   docker compose up --build
   ```

   This builds both the frontend and backend images (including downloading the SAM2 small checkpoint) and starts them together. The app will be available at `http://localhost:3000`.

3. **Stop the application:**
   ```sh
   docker compose down
   ```

> **Note:** The Docker setup runs in CPU-only mode. For GPU-accelerated segmentation, use the manual setup described above.

---

## Reverse Proxy Deployment

For local `docker compose`, the frontend builds with `VITE_BASE_PATH=/` and is served at `http://localhost:3000`.

For the shared reverse proxy deployment on `cmedia.offis.de`, build the frontend image with:

```sh
podman build --build-arg VITE_BASE_PATH=/freeze-me/ -t rm_freeze_me_frontend:latest ./frontend
```

The reverse proxy is expected to expose:

- `https://cmedia.offis.de/freeze-me/` for the frontend
- `https://cmedia.offis.de/freeze-me/backend/` for the backend API

On the server, the corresponding containers should be started on the shared proxy network as:

- `rm_freeze_me_frontend`
- `rm_freeze_me_backend`

To make the server steps repeatable, this repo now includes:

- [`scripts/build-server-images.sh`](/home/svc-cmedia/projects/freeze-me/scripts/build-server-images.sh) to build the two Podman images with the correct defaults
- [`scripts/start-server-containers.sh`](/home/svc-cmedia/projects/freeze-me/scripts/start-server-containers.sh) as a ready-to-run or ready-to-paste startup block for the server's container runner

Typical server usage:

```sh
bash scripts/build-server-images.sh
bash scripts/start-server-containers.sh
```

The shared reverse-proxy tutorial also includes a concrete Freeze Me example in [`reverse-proxy-tutorial.md`](/home/svc-cmedia/projects/reverse-proxy/reverse-proxy-tutorial.md).

---

## Useful links
- **Vue3 Introduction:** https://vuejs.org/guide/introduction.html
- **Vuetify Documentation:** https://vuetifyjs.com/en/components/buttons/#usage
- **FastAPI Examples:** https://fastapi.tiangolo.com/tutorial/first-steps/
