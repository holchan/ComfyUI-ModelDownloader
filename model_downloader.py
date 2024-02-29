import os
import requests
from urllib.parse import unquote
import comfy.sd

class ModelDownloader:
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "LINK": ("STRING", {}),
                "OUTPUT": ("STRING", {}),
            }
        }

    @staticmethod
    def load_checkpoint(LINK, OUTPUT, output_vae=True, output_clip=True):
        downloaded_file = ModelDownloader.download_model(LINK, OUTPUT)
        if downloaded_file:
            out = comfy.sd.load_checkpoint_guess_config(downloaded_file, output_vae=output_vae, output_clip=output_clip, embedding_directory=OUTPUT)
            return out[:3]
        else:
            print("Error loading checkpoint. Downloaded file not found.")
            return None

    @staticmethod
    def download_model(link, output):
        try:
            # Ensure the output directory exists
            os.makedirs(output, exist_ok=True)
            
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                # Try to get the filename from the URL
                filename = os.path.basename(unquote(link.strip('/').split('/')[-1]))
                
                # Check if the filename has a valid extension, if not, append ".dat"
                if '.' not in filename:
                    filename += ".safetensors"
                
                downloaded_file = os.path.join(output, filename)
                with open(downloaded_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                return downloaded_file
            else:
                print(f"Error downloading file: HTTP status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None

class LoRADownloader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "lora_link": ("STRING", {}),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            "output": ("STRING", {}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_link, strength_model, strength_clip, output):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        downloaded_lora_path = self.download_lora(lora_link, output)
        if downloaded_lora_path:
            downloaded_lora_content = comfy.utils.load_torch_file(downloaded_lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, downloaded_lora_content, strength_model, strength_clip)
            return (model_lora, clip_lora)
        else:
            print("Error loading Lora. Downloaded file not found.")
            return None

    def download_lora(self, link, output):
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                # Extract filename from URL
                filename = os.path.basename(unquote(link.strip('/').split('/')[-1]))
                # Ensure the output directory exists
                os.makedirs(output, exist_ok=True)
                # Save downloaded file to output directory
                downloaded_file = os.path.join(output, filename)
                with open(downloaded_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                return downloaded_file  # Return the path to the downloaded file
            else:
                print(f"Error downloading Lora file: HTTP status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading Lora file: {e}")
            return None

NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
    "LoRADownloader": LoRADownloader
}
