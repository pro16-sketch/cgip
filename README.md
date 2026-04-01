# Image Processing Demo (Flask + OpenCV)

Simple local demo that serves `index.html` and exposes `/api/process` to apply basic image operations using OpenCV.

Prereqs
- Python 3.8+
- Windows PowerShell (instructions below)

Install

Open PowerShell and run:

```powershell
python -m pip install -r requirements.txt
```

Run

```powershell
python app.py
```

Open http://localhost:5000 in your browser.

Notes
- This is a demo. Do not expose the server publicly without adding authentication and validation.
