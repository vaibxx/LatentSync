#!/bin/bash

# Set default super-resolution method to "none"
SUPERRES="none"

# Check if a super-resolution method is provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --superres) SUPERRES="$2"; shift ;;  # Capture the method (GFPGAN or CodeFormer)
    esac
    shift
done

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4" \
    --superres "$SUPERRES"  # Pass the selected super-resolution method
