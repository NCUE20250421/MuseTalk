import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs

# video_module.py
# This module handles video generation synchronized with audio.

class ModelPaths:
    """Default model paths that can be overridden"""
    UNET_MODEL = "./models/musetalkV15/unet.pth"
    UNET_CONFIG = "./models/musetalkV15/musetalk.json"
    WHISPER_MODEL = "./models/whisper"
    VAE_TYPE = "sd-vae"

def generate_video(audio_file, video_path, bbox_shift=0, extra_margin=10, 
                  model_paths=None, output_path="output_video.mp4", fps=25):
    """
    Generate lip-synced video using MuseTalk based on input audio.
    
    Args:
        audio_file (str): Path to the input audio file
        video_path (str): Path to the input video/image file or directory
        bbox_shift (int): Bounding box shift value for face detection
        extra_margin (int): Extra margin for face cropping
        model_paths (ModelPaths, optional): Model paths to use
        output_path (str, optional): Path to save the output video
        fps (int, optional): Frames per second for the output video
        
    Returns:
        str: Path to the generated video file
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video/image path not found: {video_path}")
    
    # Set default model paths if not provided
    if model_paths is None:
        model_paths = ModelPaths()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load models
        print("Loading models...")
        vae, unet, pe = load_all_model(
            unet_model_path=model_paths.UNET_MODEL,
            vae_type=model_paths.VAE_TYPE, 
            unet_config=model_paths.UNET_CONFIG,
            device=device
        )
        
        # Initialize audio processor and Whisper model
        print("Initializing audio processor...")
        audio_processor = AudioProcessor(feature_extractor_path=model_paths.WHISPER_MODEL)
        whisper = WhisperModel.from_pretrained(model_paths.WHISPER_MODEL)
        whisper = whisper.to(device=device).eval()
        
        # Extract audio features
        print("Extracting audio features...")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_file)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            whisper.dtype,
            whisper,
            librosa_length,
            fps=fps
        )
        
        # Process input images
        print("Processing input images...")
        if os.path.isfile(video_path):
            input_img_list = [video_path]
        else:
            input_img_list = [os.path.join(video_path, f) for f in os.listdir(video_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not input_img_list:
                raise ValueError(f"No image files found in directory: {video_path}")
        
        # Get face landmarks and bounding boxes
        print("Detecting face landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        # Initialize face parser
        fp = FaceParsing()
        
        # Process frames
        print("Processing frames...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            x1, y1, x2, y2 = bbox
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256))
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        # Smooth first and last frames for transition
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Generate lip-synced frames
        timesteps = torch.tensor([0], device=device)
        video_num = len(whisper_chunks)
        batch_size = 8
        
        print(f"Generating {video_num} frames...")
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            device=device
        )
        
        result_frames = []
        for whisper_batch, latent_batch in tqdm(gen):
            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            result_frames.extend(recon)
        
        # Save output video
        print(f"Saving video to {output_path}...")
        
        # Use OpenCV VideoWriter to save the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (256, 256))
        
        for frame in result_frames:
            frame = (frame * 255).astype(np.uint8)
            out.write(frame)
        
        out.release()
        print("Video generation completed successfully!")
        
        return output_path
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        raise