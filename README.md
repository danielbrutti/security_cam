# IP Camera RTSP Live Viewer

This project allows you to view live streams from IP cameras using Python and OpenCV, with real-time person and motion detection capabilities.

## Features
- Live RTSP stream viewing
- Real-time person detection using HOG (Histogram of Oriented Gradients)
- Motion detection (frame differencing)
- Visual bounding boxes for detected people and motion
- Detection counter display
- Saves images to `pictures/` when a person is detected with motion
- Auto-formatting and linting with pre-commit (black, pycln, flake8)

## Requirements
- Python 3.8+
- An IP camera with RTSP support
- [uv](https://github.com/astral-sh/uv) (modern Python package manager)

## Installation

1. Install uv if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository and navigate to the project folder.

3. Install dependencies:
   ```bash
   uv sync
   ```
   Or if you want to install the project in editable mode:
   ```bash
   uv sync --editable
   ```

4. Copy the `.env.example` file to `.env` and edit it with your camera credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your username, password and IP
   ```

## Usage

Run the main script:
```bash
uv run python src/security_cam/main.py
```

- A window will open showing the live camera feed with person and motion detection.
- Green bounding boxes will appear around detected people.
- Red bounding boxes will appear where motion is detected.
- The detection counter shows the number of people currently detected.
- Images with detected people and motion are saved in the `pictures/` folder.
- Press 'q' to close the window and exit.

## Development

To install development dependencies (testing, linting, formatting, etc.):
```bash
uv sync --group dev
```

### Auto-formatting and Linting

This project uses [pre-commit](https://pre-commit.com/) to automatically format and lint code on commit using black, pycln, and flake8.

1. Install pre-commit hooks:
   ```bash
   uv pip install --upgrade pre-commit
   pre-commit install
   ```
2. Now, every time you commit, your code will be auto-formatted and linted.

You can also run the hooks manually:
```bash
pre-commit run --all-files
```

## Project Structure

```
security_cam/
├── .env.example
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── src/
│   └── security_cam/
│       ├── __init__.py
│       └── main.py
└── pictures/
``` 