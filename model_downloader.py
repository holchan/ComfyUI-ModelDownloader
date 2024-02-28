import os
import subprocess
import comfy.sd

class ModelDownloader:
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
        wget_command = ["wget", link, "-P", output]
        try:
            subprocess.run(wget_command, check=True)
            downloaded_file = os.path.join(output, os.path.basename(link))
            return downloaded_file
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
            return None

NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}
