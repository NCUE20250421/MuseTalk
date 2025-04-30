# video_module.py
# This module handles video generation synchronized with audio using MuseTalk's native services.

import os
import sys
import subprocess
import yaml
import tempfile
from omegaconf import OmegaConf

def generate_video(audio_file, video_path, bbox_shift=0, extra_margin=10,
                   output_path="output_video.mp4", fps=25, batch_size=8,
                   unet_model_path="./models/musetalkV15/unet.pth",
                   unet_config="./models/musetalkV15/musetalk.json",
                   whisper_dir="./models/whisper",
                   version="v15"):
    """
    Generate lip-synced video using MuseTalk's native realtime_inference service.
    
    Args:
        audio_file (str): Path to the input audio file
        video_path (str): Path to the input video/image file or directory
        bbox_shift (int): Bounding box shift value for face detection
        extra_margin (int): Extra margin for face cropping
        output_path (str): Path to save the output video
        fps (int): Frames per second for the output video
        batch_size (int): Batch size for inference
        unet_model_path (str): Path to UNet model weights
        unet_config (str): Path to UNet configuration file
        whisper_dir (str): Directory containing Whisper model
        version (str): Version of MuseTalk ("v1" or "v15")
        
    Returns:
        str: Path to the generated video file
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video/image path not found: {video_path}")

    # Create a temporary YAML config file for the inference
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, "temp_inference_config.yaml")
    
    # Set avatar ID to be unique based on timestamp
    import time
    avatar_id = f"temp_avatar_{int(time.time())}"
    
    # Create directory for results
    result_dir = os.path.dirname(output_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Prepare config content
    config_content = {
        avatar_id: {
            "preparation": True,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "audio_clips": {
                "output": audio_file
            }
        }
    }
    
    # Write config to file
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_content, f)
    
    print(f"Created temporary inference config at: {temp_config_path}")
    
    # Determine output video path
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Check if ffmpeg is in PATH, if not, try to find it
    ffmpeg_path = get_ffmpeg_path()
    
    # Build command for realtime_inference.py
    cmd = [
        sys.executable,
        "-m", "scripts.realtime_inference",
        "--inference_config", temp_config_path,
        "--result_dir", output_dir,
        "--unet_model_path", unet_model_path,
        "--unet_config", unet_config,
        "--version", version,
        "--fps", str(fps),
        "--batch_size", str(batch_size),
        "--extra_margin", str(extra_margin),
        "--output_vid_name", output_name,
        "--whisper_dir", whisper_dir
    ]
    
    if ffmpeg_path:
        cmd.extend(["--ffmpeg_path", ffmpeg_path])
    
    # Run the command
    print("Running MuseTalk realtime inference...")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        
        # Determine the output file path from MuseTalk's convention
        if version == "v15":
            musetalk_output = os.path.join(output_dir, "v15", "avatars", avatar_id, "vid_output", f"{output_name}.mp4")
        else:
            musetalk_output = os.path.join(output_dir, "avatars", avatar_id, "vid_output", f"{output_name}.mp4")
        
        # Check if the output file exists
        if os.path.exists(musetalk_output):
            # Copy the file to the desired output path if different
            if musetalk_output != output_path:
                import shutil
                shutil.copy(musetalk_output, output_path)
                print(f"Video file copied from {musetalk_output} to {output_path}")
            return output_path
        else:
            # If the file wasn't found at the expected location, try to find it
            import glob
            possible_outputs = glob.glob(os.path.join(output_dir, "**", f"{output_name}.mp4"), recursive=True)
            if possible_outputs:
                if possible_outputs[0] != output_path:
                    import shutil
                    shutil.copy(possible_outputs[0], output_path)
                    print(f"Video file copied from {possible_outputs[0]} to {output_path}")
                return output_path
            else:
                raise FileNotFoundError(f"Could not find generated video file from MuseTalk")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running MuseTalk realtime inference: {e}")
        raise
    finally:
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

def get_ffmpeg_path():
    """Attempts to find the ffmpeg executable path"""
    # First check if ffmpeg is in PATH
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return None  # No need for explicit path if it's in PATH
    except:
        # Try common locations
        common_paths = [
            "ffmpeg-master-latest-win64-gpl-shared\\bin",
            "ffmpeg-4.4-amd64-static",
            "ffmpeg"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                if os.path.exists(os.path.join(path, "ffmpeg")) or os.path.exists(os.path.join(path, "ffmpeg.exe")):
                    return os.path.abspath(path)
        
        return None  # Could not find ffmpeg