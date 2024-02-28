import os
import subprocess
import folder_paths, comfy.model_management

class ModelDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LINK": ("STRING", {}),
                "OUTPUT": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, link, output, output_vae=True, output_clip=True):
        # Download the model file
        downloaded_file = self.download_model(link, output)
        if downloaded_file:
            # Load the downloaded model file
            out = comfy.sd.load_checkpoint_guess_config(downloaded_file, output_vae=True, output_clip=True, embedding_directory=output)
            return out[:3]
        else:
            print("Error loading checkpoint. Downloaded file not found.")
            return None

    def download_model(self, link, output):
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