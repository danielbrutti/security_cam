# IP Camera RTSP Live Viewer

This project allows you to view live streams from IP cameras using Python and OpenCV.

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

5. Run the viewer:
   ```bash
   uv run python src/main.py
   ```

## Development

To install development dependencies (testing, linting, etc.):
```bash
uv sync --group dev
```

To run the project with development dependencies:
```bash
uv run --group dev python src/main.py
```

## Usage
- A window will open showing the live camera feed.
- Press 'q' to close the window and exit.

## Project Structure

```
security_cam/
├── .env.example
├── pyproject.toml
├── README.md
└── src/
    └── main.py
``` 