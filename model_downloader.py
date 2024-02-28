import os
import subprocess
import folder_paths, comfy.model_management

class ModelDownloader:
    MODEL_TYPE = "Checkpoint"  # Default value

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LINK": ("STRING", {}),
                "OUTPUT": ("STRING", {}),
            }
        }

    def download_model(self, link, output, model_type=None):
        """
        Downloads a model from a given link and returns its path.

        Args:
            link (str): The URL of the model to download.
            output (str): The desired output directory for the downloaded model.
                If not specified, the current working directory will be used.
            model_type (str, optional): The model type. Defaults to None.

        Returns:
            str: The path to the downloaded model file, or None if an error occurs.
        """

        # Ensure output directory exists (create if necessary)
        os.makedirs(output, exist_ok=True)

        # Extract filename from URL or use the provided model_type
        filename = os.path.basename(link) if model_type is None else f"{model_type}.ckpt"

        # Download the model using wget (robust error handling)
        wget_command = ["wget", "-q", "-O", os.path.join(output, filename), link]
        try:
            subprocess.run(wget_command, check=True, capture_output=True)
            print(f"Downloaded model '{filename}' to '{output}'.")
            return os.path.join(output, filename)  # Return full path
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {str(e)}")
            return None

class CheckpointLoaderSimple:
    """
    Checkpoint loader class with download functionality.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name_or_link": (
                    (folder_paths.get_filename_list("checkpoints"), ),  # Local files
                    ("STRING", {}),  # Download via link
                )
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name_or_link, output_vae=True, output_clip=True):
        """
        Loads a checkpoint, either from a local file or by downloading it first.

        Args:
            ckpt_name_or_link (str): The checkpoint name (for local files)
                                       or URL (for download).
            output_vae (bool, optional): Whether to output the VAE. Defaults to True.
            output_clip (bool, optional): Whether to output the CLIP. Defaults to True.

        Returns:
            tuple: The loaded model and CLIP, or None if an error occurs.
        """

        # Check if a link is provided, download if necessary
        if isinstance(ckpt_name_or_link, str) and ckpt_name_or_link.startswith("http"):
            model_downloader = ModelDownloader()
            downloaded_file = model_downloader.download_model(ckpt_name_or_link, folder_paths.get_folder_paths("checkpoints"))
            if downloaded_file is None:
                return None  # Error occurred, return None

            # Use the downloaded file path for loading
            ckpt_path = downloaded_file
        else:
            # Use local file path directly
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name_or_link)

        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]



NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}