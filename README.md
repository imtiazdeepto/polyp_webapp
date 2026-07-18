# Polyp Segmentation CAD

Polyp Segmentation CAD is a medical image analysis app for colonoscopy images. It uses a **UNet + ResNet-18** model to detect and segment polyps, then shows the prediction as a mask, overlay, contour view, and bounding boxes.

The project has two runtime pieces:

- **FastAPI backend** for inference
- **Streamlit frontend** for image upload and visualization

The UI is styled with a light medical theme for a clean white background.

## Features

- Upload JPG or PNG colonoscopy images
- Run polyp detection and segmentation
- View confidence score and mask area
- Display original image, mask, overlay, contours, and bounding boxes
- Run locally through FastAPI + Streamlit
- Start and stop both services with a PowerShell script

## Project Structure

```text
polyp_webapp/
├── app.py              # Streamlit frontend
├── backend.py          # FastAPI inference API
├── inference.py        # Shared preprocessing / inference / postprocessing
├── model.py            # UNet + ResNet-18 model definition
├── best.pth            # Trained checkpoint
├── requirements.txt    # Python dependencies
├── manage-app.ps1      # PowerShell launcher for start/stop/status
├── .streamlit/config.toml
└── README.md
```

## Requirements

- Windows PowerShell
- Python 3.12
- A trained checkpoint named `best.pth` in the project root

## Installation

1. Open PowerShell in the project folder.

2. Create a virtual environment if you do not already have one:

```powershell
py -3.12 -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Make sure the model checkpoint is available:

```text
best.pth
```

If you want to use a different path, set `CHECKPOINT_PATH` before starting the app.

## Run Locally

### Option 1: Use the PowerShell launcher

Start the full app:

```powershell
powershell -ExecutionPolicy Bypass -File .\manage-app.ps1 start
```

Stop the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\manage-app.ps1 stop
```

Check status:

```powershell
powershell -ExecutionPolicy Bypass -File .\manage-app.ps1 status
```

### Option 2: Run the services manually

Open one terminal for the backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open a second terminal for the frontend:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

## How It Works

1. The user uploads a colonoscopy image.
2. The frontend sends the image to the FastAPI backend, or runs local inference if enabled.
3. The model predicts a binary polyp mask.
4. The app post-processes the result and returns:
   - `tumor_present`
   - `confidence`
   - `mask_area_px`
   - `num_regions`
   - `bboxes`
   - rendered images as base64 PNG strings

## API Endpoints

### `GET /`

Health check for the backend.

### `POST /predict`

Upload a JPG or PNG image and receive segmentation results.

Required form field:

- `file` — image file

Response fields:

- `tumor_present` — boolean
- `confidence` — prediction confidence
- `mask_area_px` — number of predicted mask pixels
- `num_regions` — number of detected regions
- `bboxes` — bounding box coordinates
- `original_image` — base64 PNG
- `mask_image` — base64 PNG
- `overlay_image` — base64 PNG
- `contour_image` — base64 PNG
- `bbox_image` — base64 PNG

## Environment Variables

| Variable           | Default                         | Purpose                                      |
| ------------------ | ------------------------------- | -------------------------------------------- |
| `CHECKPOINT_PATH`  | `best.pth`                      | Path to the model checkpoint                 |
| `CHECKPOINT_URL`   | empty                           | Optional download URL for the checkpoint     |
| `MAX_FILE_SIZE_MB` | `10`                            | Maximum upload size                          |
| `API_URL`          | `http://localhost:8000/predict` | Backend URL used by Streamlit                |
| `BACKEND_HOST`     | empty                           | Overrides `API_URL` for hosted deployments   |
| `USE_LOCAL_MODEL`  | `0`                             | Set to `1` to run inference inside Streamlit |

## Troubleshooting

- If the app says the checkpoint is missing, confirm `best.pth` exists in the project root.
- If the frontend cannot connect, make sure the backend is running on port `8000`.
- If Streamlit shows an image keyword error, update the app and use the current `app.py` in this repository.
- If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

## Notes

- The app uses a light, white medical-style background.
- The PowerShell launcher writes a small `.app-pids.json` file so it can stop the correct processes later.
- Large model files should not be committed unless you intentionally want them in Git history.

## License

Add your preferred license here before publishing to GitHub.
