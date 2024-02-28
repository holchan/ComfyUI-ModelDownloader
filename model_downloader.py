import os
import subprocess
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
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                # Try to get the filename from the Content-Disposition header
                if 'Content-Disposition' in response.headers:
                    disposition = response.headers['Content-Disposition']
                    filename = disposition.split('filename=')[1]
                    filename = unquote(filename).strip('"')
                else:
                    # If the filename is not provided, use a default name
                    filename = os.path.basename(link)
                
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

NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}