import os
import subprocess

class ModelDownloader:
    MODEL_TYPE = "Checkpoint"  # Default value
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LINK": ("STRING", {}),
                "OUTPUT": ("STRING", {}),
                "MODEL_TYPE": (["checkpoint", "lora"], ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE") if "checkpoint" in MODEL_TYPE else ("MODEL", "CLIP")
    FUNCTION = "download_model"
    CATEGORY = "advanced/loaders"

    def download_model(self, link, output, model_type):
        # Download the model using wget
        wget_command = ["wget", link, "-P", output]
        try:
            subprocess.run(wget_command, check=True)
            downloaded_file = os.path.join(output, os.path.basename(link))
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {str(e)}")
            return None

        # Load the downloaded model based on its type
        if model_type == "checkpoint":
            return comfy.sd.load_checkpoint(downloaded_file)
        elif model_type == "lora":
            raise NotImplementedError("Loading LoRA from downloaded file not yet implemented")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}