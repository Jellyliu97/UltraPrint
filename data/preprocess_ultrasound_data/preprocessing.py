import os
import glob
import numpy as np
import librosa
import cv2
import soundfile as sf
from tqdm import tqdm
import shutil
import zipfile
import tempfile

outer_folder_lists = ['20260121', '20260122', '20260123', '20260124']

def extract_rawdata2data(raw_input_root, raw_output_root):
    if not os.path.exists(raw_output_root):
        os.makedirs(raw_output_root)

    # Iterate through outer folders in raw_input_root
    for outer_folder in tqdm(os.listdir(raw_input_root), desc="Processing Folders"):
        if outer_folder not in outer_folder_lists:  # Skip unwanted folders
            continue
        outer_path = os.path.join(raw_input_root, outer_folder)
        
        # Ensure it is a directory
        if not os.path.isdir(outer_path):
            continue

        # Iterate through items in the outer folder
        for item in os.listdir(outer_path):
            # Check if item ends with .zip and is a directory
            print("extract:", item)
            if item.endswith('.zip') and item[0] != '.':
                item_path = os.path.join(outer_path, item)
                #读取.zip文件中的内容，并保存到raw_output_root对应的文件夹中
                # Create a temporary directory for extraction
                # This directory and its contents are automatically deleted when the block exits
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        with zipfile.ZipFile(item_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        
                        # Prepare the target directory (folder name matches zip file name)
                        target_dir = os.path.join(raw_output_root, item[:-4])
                        os.makedirs(target_dir, exist_ok=True)

                        # Recursively walk through the temp directory and move files
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                src_file = os.path.join(root, file)
                                # Flatten structure: save directly to target_dir
                                dst_file = os.path.join(target_dir, file)
                                shutil.copy2(src_file, dst_file)
                                
                    except zipfile.BadZipFile:
                        print(f"Warning: Could not unzip {item_path}")

def extract_distance_unchanged(data_I, data_Q):
    # 0. Time Gain Compensation (TGC)
    # Compensate for signal attenuation over time (depth) using an exponential gain.
    points = data_I.shape[-1]
    indices = np.arange(points)
    
    # Attenuation coefficient (alpha). 
    # Adjust this value based on the medium's attenuation and sampling parameters.
    # For example, with 3000 points, alpha=0.0005 gives approx 4.5x gain at the end.
    alpha = 0.0005  
    tgc_gain = np.exp(alpha * indices)
    
    # Apply gain to I and Q channels
    data_I = data_I * tgc_gain
    data_Q = data_Q * tgc_gain




    # 1. Eliminate DC component (Static Clutter Removal)
    # Subtract the mean along the time axis to remove static reflections
    mean_I = np.mean(data_I, axis=-1, keepdims=True)
    mean_Q = np.mean(data_Q, axis=-1, keepdims=True)
    
    I_no_dc = data_I - mean_I
    Q_no_dc = data_Q - mean_Q

    # 2. Extract distance-independent features
    # Normalize by the instantaneous magnitude to remove amplitude attenuation caused by distance.
    # This retains the phase/Doppler information which characterizes the target's motion/structure.
    magnitude = np.sqrt(I_no_dc**2 + Q_no_dc**2)
    
    # Avoid division by zero
    magnitude[magnitude < 1e-8] = 1.0
    
    # Element-wise normalization
    I_norm = I_no_dc / magnitude
    Q_norm = Q_no_dc / magnitude
    
    return I_norm, Q_norm

def process_ultrasound_iq(ultrasound_data):
    """
    Process ultrasound data: 
    1. Input shape: (16, 2, points) -> 16 frequencies, 2 record channels (Q + j I per channel), time points.
    2. Convert I/Q to magnitude: sqrt(I^2 + Q^2).
    3. Normalize each frequency channel independently.
    4. Output shape: (16, points).
    """
    # print("ultrasound_data 类型:", ultrasound_data.dtype)
    # print("ultrasound_data 形状:", ultrasound_data.shape)
    # print("ultrasound_data 示例值:", ultrasound_data.flatten()[:100])
    # ultrasound_data = ultrasound_data.astype(np.float64)

    # Calculate magnitude from I/Q (shape: 16, points)
    # magnitude = np.sqrt(ultrasound_data[:, 0, :].real**2 + ultrasound_data[:, 0, :].imag**2)
    # phase = np.arctan2(ultrasound_data[:, 0, :].imag, ultrasound_data[:, 0, :].real)
    # phase = np.unwrap(phase, axis=1)

    cal = ultrasound_data
    # # Normalize each frequency channel
    # normalized_data = np.zeros_like(cal)
    # for i in range(cal.shape[0]):
    #     channel_data = cal[i, :]
    #     min_val = np.min(channel_data)
    #     max_val = np.max(channel_data)
    #     if max_val - min_val > 1e-8:
    #         normalized_data[i, :] = (channel_data - min_val) / (max_val - min_val)
    #     else:
    #         normalized_data[i, :] = channel_data
    # Transpose from (16, 2, T) to (2, 16, T) -> (Channel, Frequency, Time)


    #对每一个segment的IQ数据进行归一化处理,然后和不同的频点堆叠在一起
    data_transposed = cal.transpose(1, 0, 2)
    
    # Extract I (Real) and Q (Imag) components
    data_I = data_transposed.real
    data_Q = data_transposed.imag

    # data_I, data_Q = extract_distance_unchanged(data_I, data_Q)

    # Normalize I component (Min-Max normalization along time axis)
    min_I = np.min(data_I, axis=-1, keepdims=True)
    max_I = np.max(data_I, axis=-1, keepdims=True)
    range_I = max_I - min_I
    range_I[range_I < 1e-8] = 1.0  # Avoid division by zero
    norm_I = (data_I - min_I) / range_I

    # Normalize Q component (Min-Max normalization along time axis)
    min_Q = np.min(data_Q, axis=-1, keepdims=True)
    max_Q = np.max(data_Q, axis=-1, keepdims=True)
    range_Q = max_Q - min_Q
    range_Q[range_Q < 1e-8] = 1.0  # Avoid division by zero
    norm_Q = (data_Q - min_Q) / range_Q

    # Stack I and Q to shape (2, 16, 2, T)
    combined = np.stack((norm_I, norm_Q), axis=2)

    # Reshape to (2, 32, T), merging frequency (16) and IQ (2) dimensions
    normalized_data = combined.reshape(2, 32, -1)
            
    return normalized_data

def save_video_segment(video_path, start_sec, duration_sec, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 29.8 # Fallback
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    start_frame = int(start_sec * fps)
    total_frames = int(duration_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_written = 0
    while frames_written < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        
    cap.release()
    out.release()

def preprocess_dataset(input_root, output_root, T=1, stride=1):
    """
    Args:
        input_root: Path to raw data.
        output_root: Path to save dataset.
        T: Window length in seconds.
        stride: Sliding window stride in seconds.
    """
    # Parameters
    AUDIO_SR = 48000
    US_SR = 3000
    
    user_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    # skip_users = ['pmz', 'ldx', 'syk', 'db', 'ldx0', 'whh', 'zyc']
    # user_dirs = [u for u in user_dirs if u not in skip_users]

    for user in tqdm(user_dirs, desc="Processing Users"):
        
        user_input_path = os.path.join(input_root, user)
        user_output_path = os.path.join(output_root, user)
        os.makedirs(user_output_path, exist_ok=True)
        
        # Find files
        wav_files = glob.glob(os.path.join(user_input_path, "*.wav"))
        mp4_files = glob.glob(os.path.join(user_input_path, "*.mp4"))
        npy_files = glob.glob(os.path.join(user_input_path, "*.npy"))
        
        if not (wav_files and mp4_files and npy_files):
            print(f"Skipping {user}: Missing files.")
            continue
            
        # Assuming one file per type per user, or taking the first match
        wav_path = wav_files[0]
        mp4_path = mp4_files[0]
        npy_path = npy_files[0]
        
        # Load Data
        audio_data, _ = librosa.load(wav_path, sr=AUDIO_SR)
        raw_us_data = np.load(npy_path) # Expected shape (16, 2, N)
        
        # Process Ultrasound IQ -> Magnitude -> Normalization
        # processed_us_data = process_ultrasound_iq(raw_us_data) # Shape (16, N)
        
        # Determine duration based on shortest stream to avoid index errors
        # (Though prompt implies equal length)
        audio_duration = len(audio_data) / AUDIO_SR
        us_duration = raw_us_data.shape[-1] / US_SR
        total_duration = min(audio_duration, us_duration)
        
        # Sliding Window Segmentation
        current_time = 0.0
        sample_idx = 0
        
        while current_time + T <= total_duration:
            segment_name = f"{user}_seg_{sample_idx:04d}"
            
            # 1. Slice and Save Ultrasound (16 channels)
            us_start_idx = int(current_time * US_SR)
            us_end_idx = int((current_time + T) * US_SR)
            us_segment = raw_us_data[:, :, us_start_idx:us_end_idx]
            
            processed_us = process_ultrasound_iq(us_segment)
            us_save_path = os.path.join(user_output_path, f"{segment_name}_us.npy")
            np.save(us_save_path, processed_us)
            
            # 2. Slice and Save Audio
            audio_start_idx = int(current_time * AUDIO_SR)
            audio_end_idx = int((current_time + T) * AUDIO_SR)
            audio_segment = audio_data[audio_start_idx:audio_end_idx]
            
            audio_save_path = os.path.join(user_output_path, f"{segment_name}_audio.wav")
            sf.write(audio_save_path, audio_segment, AUDIO_SR)
            
            # 3. Slice and Save Video
            video_save_path = os.path.join(user_output_path, f"{segment_name}_video.mp4")
            save_video_segment(mp4_path, current_time, T, video_save_path)
            
            current_time += stride
            sample_idx += 1

def preprocess_dataset2(input_root, output_root, T=1, stride=1):
        select_folder = ["impact_angle", "impact_dis", "impact_phone"]
        # Parameters
        AUDIO_SR = 48000
        US_SR = 3000

        # Iterate over top-level folders (Group/Specific Folders) in input_root
        for group_folder in tqdm(os.listdir(input_root), desc="Processing Groups"):
            if group_folder not in select_folder: ###select specific folders
                continue
            group_path = os.path.join(input_root, group_folder)
            if not os.path.isdir(group_path):
                continue

            # Output to a folder with the same name in output_root
            # All samples for this group (from inner actions/users) will be flattened here
            group_output_path = os.path.join(output_root, group_folder)
            os.makedirs(group_output_path, exist_ok=True)

            # Layer 1: Iterate Action folders
            for action_folder in os.listdir(group_path):
                action_path = os.path.join(group_path, action_folder)
                if not os.path.isdir(action_path):
                    continue
                
                # Layer 2: Iterate User folders
                for user_folder in os.listdir(action_path):
                    user_path = os.path.join(action_path, user_folder)
                    if not os.path.isdir(user_path):
                        continue

                    # Locate files
                    wav_files = glob.glob(os.path.join(user_path, "*.wav"))
                    mp4_files = glob.glob(os.path.join(user_path, "*.mp4"))
                    npy_files = glob.glob(os.path.join(user_path, "*.npy"))
                    
                    if not (wav_files and mp4_files and npy_files):
                        continue
                        
                    wav_path = wav_files[0]
                    mp4_path = mp4_files[0]
                    npy_path = npy_files[0]
                    
                    try:
                        # Load Data
                        audio_data, _ = librosa.load(wav_path, sr=AUDIO_SR)
                        raw_us_data = np.load(npy_path)
                        
                        audio_duration = len(audio_data) / AUDIO_SR
                        us_duration = raw_us_data.shape[-1] / US_SR
                        total_duration = min(audio_duration, us_duration)
                        
                        current_time = 0.0
                        sample_idx = 0
                        
                        # Prefix format: Group_Action_User
                        prefix = f"{group_folder}_{action_folder}_{user_folder}"
                        
                        while current_time + T <= total_duration:
                            segment_name = f"{prefix}_seg_{sample_idx:04d}"
                            
                            # 1. Ultrasound Processing
                            us_start = int(current_time * US_SR)
                            us_end = int((current_time + T) * US_SR)
                            us_seg = raw_us_data[:, :, us_start:us_end]
                            proc_us = process_ultrasound_iq(us_seg)
                            
                            np.save(os.path.join(group_output_path, f"{segment_name}_us.npy"), proc_us)
                            
                            # 2. Audio Processing
                            au_start = int(current_time * AUDIO_SR)
                            au_end = int((current_time + T) * AUDIO_SR)
                            au_seg = audio_data[au_start:au_end]
                            sf.write(os.path.join(group_output_path, f"{segment_name}_audio.wav"), au_seg, AUDIO_SR)
                            
                            # 3. Video Processing
                            save_video_segment(mp4_path, current_time, T, os.path.join(group_output_path, f"{segment_name}_video.mp4"))
                            
                            current_time += stride
                            sample_idx += 1
                            
                    except Exception as e:
                        print(f"Error processing files in {user_path}: {e}")

        # Explicitly creating a comment here to absorb the trailing function signature in the file
        #
    


def move_dataset_npy(OUTPUT_DIR, OUTPUT_NPY_DIR):
        if not os.path.exists(OUTPUT_NPY_DIR):
            os.makedirs(OUTPUT_NPY_DIR)

        # Iterate through each user folder in the output directory
        for user_folder in tqdm(os.listdir(OUTPUT_DIR), desc="Moving NPY Files"):
            user_path = os.path.join(OUTPUT_DIR, user_folder)
            
            # Ensure it is a directory
            if not os.path.isdir(user_path):
                continue

            # Create corresponding user folder in the NPY destination directory
            target_user_path = os.path.join(OUTPUT_NPY_DIR, user_folder)
            os.makedirs(target_user_path, exist_ok=True)

            # Iterate through files in the user folder
            for item in os.listdir(user_path):
                if item.endswith('.npy'):
                    src_file = os.path.join(user_path, item)
                    dst_file = os.path.join(target_user_path, item)
                    shutil.copy2(src_file, dst_file)

def move_dataset_npy2(OUTPUT_DIR, OUTPUT_NPY_DIR):
        if not os.path.exists(OUTPUT_NPY_DIR):
            os.makedirs(OUTPUT_NPY_DIR)
        select_folder = ["impact_angle", "impact_dis", "impact_phone"]

        # Iterate through user folders
        for user_folder in tqdm(os.listdir(OUTPUT_DIR), desc="Moving Paired NPY Files"):
            if user_folder not in select_folder: ###select specific folders
                continue
            user_path = os.path.join(OUTPUT_DIR, user_folder)
            
            if not os.path.isdir(user_path):
                continue

            target_user_path = os.path.join(OUTPUT_NPY_DIR, user_folder)
            os.makedirs(target_user_path, exist_ok=True)

            files = os.listdir(user_path)
            files_set = set(files)

            for file in files:
                # Check for _us.npy files
                if file.endswith('_us.npy'):
                    prefix = file[:-7] # Remove '_us.npy'
                    video_file = f"{prefix}_video.npy"

                    # Check if corresponding _video.npy exists
                    if video_file in files_set:
                        src_us = os.path.join(user_path, file)
                        src_video = os.path.join(user_path, video_file)

                        dst_us = os.path.join(target_user_path, file)
                        dst_video = os.path.join(target_user_path, video_file)

                        # Move (Copy) both files
                        shutil.copy2(src_us, dst_us)
                        shutil.copy2(src_video, dst_video)
        #


# def rename_path():
    

if __name__ == "__main__":
    RAW_INPUT_DIR = r"E:\dataset\ultrasound_video_audio\DATA_RAW\RAW_DATA"
    INPUT_DIR = r"E:\dataset\ultrasound_video_audio\DATA\raw_data"
    OUTPUT_DIR = r"E:\dataset\ultrasound_video_audio\DATA\dataset"
    OUTPUT_NPY_DIR = r"E:\dataset\ultrasound_video_audio\DATA\dataset_npy"


    # extract_rawdata2data(RAW_INPUT_DIR, INPUT_DIR)

    # Set window length T and stride here
    WINDOW_LENGTH_T = 5  # Example: T=5 seconds 
    STRIDE = 5           # 1 second stride
    
    print("Starting preprocessing...")
    # preprocess_dataset(INPUT_DIR, OUTPUT_DIR, T=WINDOW_LENGTH_T, stride=STRIDE)
    # preprocess_dataset2(INPUT_DIR, OUTPUT_DIR, T=WINDOW_LENGTH_T, stride=STRIDE)
    print("Preprocessing complete.")


    move_dataset_npy2(OUTPUT_DIR, OUTPUT_NPY_DIR)
    
    