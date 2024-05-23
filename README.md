# ComfyUI_IPAdapter_plus_model_download

a script to download the ComfyUI_IPAdapter_plus models

## Features

- Downloads models for different categories (clip_vision, ipadapter, loras).
- Supports concurrent downloads to save time.
- Displays download progress using a progress bar.
- Automatically creates necessary directories if they do not exist.

## Requirements

- Python 3.6 or later
- `requests` library
- `tqdm` library

```sh
pip install requests tqdm
```

Usage

```shell
python download_models.py /path/to/ComfyUI/models --max_concurrent_downloads 3

```
