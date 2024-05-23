import os
import requests
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# List of models to download with their URLs and target paths
models = {
    "clip_vision": [
        ("CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"),
        ("CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors")
    ],
    "ipadapter": [
        ("ip-adapter_sd15.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"),
        ("ip-adapter_sd15_light_v11.bin",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin"),
        ("ip-adapter-plus_sd15.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"),
        ("ip-adapter-plus-face_sd15.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors"),
        ("ip-adapter-full-face_sd15.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"),
        ("ip-adapter_sd15_vit-G.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"),
        ("ip-adapter_sdxl_vit-h.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"),
        ("ip-adapter-plus_sdxl_vit-h.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"),
        ("ip-adapter-plus-face_sdxl_vit-h.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ("ip-adapter_sdxl.safetensors",
         "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"),

        ("ip-adapter-faceid_sd15.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"),
        ("ip-adapter-faceid-plusv2_sd15.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin"),
        ("ip-adapter-faceid-portrait-v11_sd15.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin"),
        ("ip-adapter-faceid_sdxl.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin"),
        ("ip-adapter-faceid-plusv2_sdxl.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin"),
        ("ip-adapter-faceid-portrait_sdxl.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin"),
        ("ip-adapter-faceid-portrait_sdxl_unnorm.bin",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl_unnorm.bin")
    ],
    "loras": [
        ("ip-adapter-faceid_sd15_lora.safetensors",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors"),
        ("ip-adapter-faceid-plusv2_sd15_lora.safetensors",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"),
        ("ip-adapter-faceid_sdxl_lora.safetensors",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors"),
        ("ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
         "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors")
    ]
}


def download_file(url, path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(path, 'wb') as file, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_model(sub_dir, model_name, url, model_dir):
    full_path = os.path.join(model_dir, sub_dir, model_name)
    if not os.path.exists(full_path):
        print(f"Downloading {model_name} to {full_path}")
        download_file(url, full_path)
    else:
        print(f"{model_name} already exists in {
              full_path}, skipping download.")


def main(model_dir, max_concurrent_downloads):
    for sub_dir in models.keys():
        full_path = os.path.join(model_dir, sub_dir)
        os.makedirs(full_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as executor:
        futures = []
        for sub_dir, model_list in models.items():
            for model_name, url in model_list:
                futures.append(executor.submit(
                    download_model, sub_dir, model_name, url, model_dir))

        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download models for ComfyUI to specified directory.")
    parser.add_argument("model_dir", type=str,
                        help="The root directory to download the models to.")
    parser.add_argument("--max_concurrent_downloads", type=int,
                        default=3, help="Maximum number of concurrent downloads.")
    args = parser.parse_args()
    main(args.model_dir, args.max_concurrent_downloads)
