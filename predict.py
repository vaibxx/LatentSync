# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet  # For GFPGAN
from gfpgan import GFPGANer
from facelib.utils.face_restoration_helper import FaceRestoreHelper  # For CodeFormer

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio to ", default=None),
        guidance_scale: float = Input(description="Guidance scale", ge=0, le=10, default=1.0),
        seed: int = Input(description="Set to 0 for Random seed", default=0),
        superres: str = Input(description="Super-resolution method [none/GFPGAN/CodeFormer]", default="none"),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/second_stage.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "/tmp/video_out.mp4"

        # Run the LatentSync model
        os.system(f"python -m scripts.inference --unet_config_path {config_path} --inference_ckpt_path {ckpt_path} --guidance_scale {str(guidance_scale)} --video_path {video_path} --audio_path {audio_path} --video_out_path {output_path} --seed {seed} --superres {superres}")

        # Apply Super-Resolution (if enabled)
        if superres.lower() in ["gfpgan", "codeformer"]:
            output_path = self.apply_super_resolution(output_path, superres)

        return Path(output_path)

    def apply_super_resolution(self, video_path, method):
        """Apply GFPGAN or CodeFormer to the generated lipsynced region if needed"""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out_path = "/tmp/video_superres.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        if method == "gfpgan":
            face_enhancer = GFPGANer(model_path="checkpoints/GFPGANv1.4.pth", upscale=2)
        elif method == "codeformer":
            face_enhancer = FaceRestoreHelper(upscale=2)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply super-resolution to the generated region
            enhanced_frame = face_enhancer.enhance(frame, paste_back=True)[0]

            out.write(enhanced_frame)

        cap.release()
        out.release()
        return out_path
