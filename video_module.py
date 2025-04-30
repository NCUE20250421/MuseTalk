# video_module.py
# This module handles video generation synchronized with audio using MuseTalk's native services.

import os
import sys
import subprocess
import yaml
import tempfile
import torch
import numpy as np
from omegaconf import OmegaConf
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing

# 全局模型變量
global_models = {
    "loaded": False,
    "vae": None,
    "unet": None,
    "pe": None,
    "whisper": None,
    "audio_processor": None,
    "face_parser": None,
    "device": None,
    "weight_dtype": None,
    "config": {
        "unet_model_path": "./models/musetalkV15/unet.pth",
        "unet_config": "./models/musetalkV15/musetalk.json",
        "whisper_dir": "./models/whisper",
        "vae_type": "sd-vae",
        "version": "v15",
        "extra_margin": 10,
        "left_cheek_width": 90,
        "right_cheek_width": 90
    }
}

def preload_models(unet_model_path=None, unet_config=None, whisper_dir=None, 
                 version="v15", vae_type="sd-vae", gpu_id=0, 
                 extra_margin=10, left_cheek_width=90, right_cheek_width=90):
    """
    預先載入 MuseTalk 模型，避免每次處理時重新載入
    
    Args:
        unet_model_path (str): UNet 模型路徑
        unet_config (str): UNet 配置文件路徑
        whisper_dir (str): Whisper 模型目錄
        version (str): MuseTalk 版本 ("v1" 或 "v15")
        vae_type (str): VAE 類型
        gpu_id (int): GPU ID
        extra_margin (int): 面部裁剪額外邊距
        left_cheek_width (int): 左臉頰寬度
        right_cheek_width (int): 右臉頰寬度
        
    Returns:
        bool: 加載成功返回 True
    """
    global global_models
    
    # 如果已經載入過模型，直接返回
    if global_models["loaded"]:
        print("Models already loaded.")
        return True
    
    # 更新配置
    if unet_model_path:
        global_models["config"]["unet_model_path"] = unet_model_path
    if unet_config:
        global_models["config"]["unet_config"] = unet_config
    if whisper_dir:
        global_models["config"]["whisper_dir"] = whisper_dir
    global_models["config"]["version"] = version
    global_models["config"]["vae_type"] = vae_type
    global_models["config"]["extra_margin"] = extra_margin
    global_models["config"]["left_cheek_width"] = left_cheek_width
    global_models["config"]["right_cheek_width"] = right_cheek_width
    
    print("Preloading MuseTalk models...")
    
    try:
        # 設置計算設備
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        global_models["device"] = device
        print(f"Using device: {device}")
        
        # 載入模型
        vae, unet, pe = load_all_model(
            unet_model_path=global_models["config"]["unet_model_path"],
            vae_type=global_models["config"]["vae_type"],
            unet_config=global_models["config"]["unet_config"],
            device=device
        )
        
        # 將模型轉為半精度
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        
        # 初始化音頻處理器和 Whisper 模型
        audio_processor = AudioProcessor(feature_extractor_path=global_models["config"]["whisper_dir"])
        weight_dtype = unet.model.dtype
        whisper = WhisperModel.from_pretrained(global_models["config"]["whisper_dir"])
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        
        # 初始化面部解析器
        if version == "v15":
            face_parser = FaceParsing(
                left_cheek_width=global_models["config"]["left_cheek_width"],
                right_cheek_width=global_models["config"]["right_cheek_width"]
            )
        else:  # v1
            face_parser = FaceParsing()
        
        # 保存模型到全局變量
        global_models["vae"] = vae
        global_models["unet"] = unet
        global_models["pe"] = pe
        global_models["whisper"] = whisper
        global_models["audio_processor"] = audio_processor
        global_models["face_parser"] = face_parser
        global_models["weight_dtype"] = weight_dtype
        global_models["loaded"] = True
        
        print("MuseTalk models preloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error preloading models: {str(e)}")
        global_models["loaded"] = False
        return False

def generate_video(audio_file, video_path, bbox_shift=0, extra_margin=10,
                   output_path="output_video.mp4", fps=25, batch_size=8,
                   unet_model_path=None, unet_config=None, whisper_dir=None,
                   version="v15", use_preloaded_models=False):
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
        use_preloaded_models (bool): Whether to use preloaded models
        
    Returns:
        str: Path to the generated video file
    """
    global global_models
    
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video/image path not found: {video_path}")
    
    # 如果需要使用預先載入的模型但模型尚未載入，先載入模型
    if use_preloaded_models and not global_models["loaded"]:
        preload_models(
            unet_model_path=unet_model_path,
            unet_config=unet_config,
            whisper_dir=whisper_dir,
            version=version,
            extra_margin=extra_margin
        )
    
    # 如果已預先載入模型且選擇使用，則直接使用內存中的模型
    if use_preloaded_models and global_models["loaded"]:
        print("Using preloaded models for video generation...")
        return _generate_video_with_preloaded_models(
            audio_file=audio_file,
            video_path=video_path,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            output_path=output_path,
            fps=fps,
            batch_size=batch_size
        )
    
    # 否則，使用腳本調用方式生成視頻
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
    
    # Use provided paths or default paths
    if not unet_model_path:
        unet_model_path = global_models["config"]["unet_model_path"]
    if not unet_config:
        unet_config = global_models["config"]["unet_config"]
    if not whisper_dir:
        whisper_dir = global_models["config"]["whisper_dir"]
    
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

def _generate_video_with_preloaded_models(audio_file, video_path, bbox_shift=0, extra_margin=10,
                                         output_path="output_video.mp4", fps=25, batch_size=8):
    """
    使用預先載入的模型生成視頻（內部實現）
    """
    global global_models
    
    try:
        import os
        import cv2
        import glob
        import pickle
        import torch
        import numpy as np
        from tqdm import tqdm
        from musetalk.utils.utils import datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from musetalk.utils.blending import get_image_prepare_material, get_image_blending
        
        # 創建輸出目錄
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 獲取預加載的模型和配置
        device = global_models["device"]
        vae = global_models["vae"]
        unet = global_models["unet"]
        pe = global_models["pe"]
        whisper = global_models["whisper"]
        audio_processor = global_models["audio_processor"]
        fp = global_models["face_parser"]
        weight_dtype = global_models["weight_dtype"]
        
        # 設置時間步長
        timesteps = torch.tensor([0], device=device)
        
        print("Extracting audio features...")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_file, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        
        # 處理輸入圖像
        print("Processing input images...")
        if os.path.isfile(video_path):
            input_img_list = [video_path]
        else:
            input_img_list = [os.path.join(video_path, f) for f in os.listdir(video_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not input_img_list:
                raise ValueError(f"No image files found in directory: {video_path}")
        
        # 獲取人臉關鍵點和邊界框
        print("Detecting face landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        # 處理圖像框
        print("Processing frames...")
        input_latent_list = []
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            # 跳過無效的邊界框
            if bbox == (0.0, 0.0, 0.0, 0.0):
                continue
            
            x1, y1, x2, y2 = bbox
            # 添加額外邊距
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]  # 更新 bbox
            
            # 裁剪並調整大小
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 獲取潛在表示
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # 為平滑過渡，添加反向順序的框
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # 生成視頻幀
        video_num = len(whisper_chunks)
        print(f"Generating {video_num} frames...")
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            device=device
        )
        
        # 創建臨時目錄存放中間結果
        temp_dir = tempfile.mkdtemp()
        try:
            # 處理生成的幀
            mask_list = []
            mask_coords_list = []
            
            # 為每個輸入幀準備遮罩
            for i, frame in enumerate(frame_list):
                x1, y1, x2, y2 = coord_list[i]
                mask, crop_box = get_image_prepare_material(
                    frame, [x1, y1, x2, y2], fp=fp, mode="jaw"
                )
                mask_list.append(mask)
                mask_coords_list.append(crop_box)
            
            # 循環處理生成的幀
            result_frames = []
            idx = 0
            
            for whisper_batch, latent_batch in tqdm(gen):
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(dtype=unet.model.dtype, device=device)
                
                # 預測潛在表示
                pred_latents = unet.model(
                    latent_batch, 
                    timesteps, 
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                # 解碼潛在表示為圖像
                recon_frames = vae.decode_latents(pred_latents)
                
                # 將生成的幀與原始幀混合
                for res_frame in recon_frames:
                    bbox = coord_list[idx % len(coord_list)]
                    ori_frame = frame_list[idx % len(frame_list)].copy()
                    x1, y1, x2, y2 = bbox
                    
                    # 調整大小
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    
                    mask = mask_list[idx % len(mask_list)]
                    mask_crop_box = mask_coords_list[idx % len(mask_coords_list)]
                    
                    # 混合幀
                    combine_frame = get_image_blending(
                        ori_frame, res_frame, bbox, mask, mask_crop_box
                    )
                    
                    result_frames.append(combine_frame)
                    
                    # 保存中間幀
                    cv2.imwrite(
                        os.path.join(temp_dir, f"{idx:08d}.png"), 
                        combine_frame
                    )
                    
                    idx += 1
            
            # 使用 OpenCV VideoWriter 保存視頻
            print(f"Saving video to {output_path}...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'avc1'
            out = cv2.VideoWriter(
                os.path.join(temp_dir, "temp_video.mp4"), 
                fourcc, 
                float(fps), 
                (frame_list[0].shape[1], frame_list[0].shape[0])
            )
            
            for frame_path in sorted(glob.glob(os.path.join(temp_dir, "*.png"))):
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            
            # 使用 FFmpeg 添加音頻
            ffmpeg_path = get_ffmpeg_path()
            ffmpeg_cmd = "ffmpeg"
            if ffmpeg_path:
                if os.path.exists(os.path.join(ffmpeg_path, "ffmpeg.exe")):
                    ffmpeg_cmd = os.path.join(ffmpeg_path, "ffmpeg.exe")
                elif os.path.exists(os.path.join(ffmpeg_path, "ffmpeg")):
                    ffmpeg_cmd = os.path.join(ffmpeg_path, "ffmpeg")
            
            cmd = [
                ffmpeg_cmd, "-y", "-i", 
                os.path.join(temp_dir, "temp_video.mp4"), 
                "-i", audio_file, 
                "-c:v", "copy", "-c:a", "aac", 
                "-strict", "experimental", 
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Video generation completed successfully: {output_path}")
            return output_path
        
        finally:
            # 清理臨時文件
            import shutil
            shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"Error in video generation with preloaded models: {str(e)}")
        raise

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