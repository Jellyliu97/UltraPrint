import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_csv(root_dir, train_ratio=0.8):
    # Lists to store dataset information
    all_experiments = []
    
    # Get all user directories
    user_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    skip_users = ['ldx0', 'ldx1', 'ldx2', 'ldx3', 'ldx4', 'ldx5']
    user_dirs = [u for u in user_dirs if u not in skip_users]

    # Create a map for user labels (index from 0)
    user_to_idx = {user: idx for idx, user in enumerate(sorted(user_dirs))}
    
    print(f"Found {len(user_dirs)} users.")
    print(f"User mapping: {user_to_idx}")

    for user in user_dirs:
        user_path = os.path.join(root_dir, user)
        
        # Find all .mp4 files to identify base valid samples
        # Assuming format:  {name}_video.mp4
        video_files = glob.glob(os.path.join(user_path, "*_video.mp4"))
        
        for vid_path in video_files:
            # Extract base name to verify other files exist
            # e.g., C:/.../User1/sample1_video.mp4 -> sample1
            base_name_with_path = vid_path.replace('_video.mp4', '')
            base_name = os.path.basename(base_name_with_path)
            
            audio_path = f"{base_name_with_path}_audio.wav"
            ultrasound_path = f"{base_name_with_path}_us.npy"
            
            video_feature_path = f"{base_name_with_path}_video.npy"
            
            # Check if corresponding files exist
            if os.path.exists(audio_path) and os.path.exists(ultrasound_path) and os.path.exists(video_feature_path):
                sample_info = {
                    'user_name': user,
                    'label': user_to_idx[user],
                    'video_path': vid_path,
                    'audio_path': audio_path,
                    'ultrasound_path': ultrasound_path,
                    'sample_name': base_name,
                    'video_feature_path': video_feature_path,
                    'TrueUser': 1
                }
                all_experiments.append(sample_info)
            else:
                print(f"Warning: Missing files for {vid_path}, skipping.")

    if not all_experiments:
        print("No valid samples found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_experiments)
    
    # Shuffle and split the dataset
    # Stratify by 'label' ensures equal distribution of users in train and test
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=(1 - train_ratio), 
            stratify=df['label'], 
            random_state=42
        )
    except ValueError as e:
        # Fallback if a class has too few members to stratify
        print(f"Stratification failed (likely single sample class), performing random split. Error: {e}")
        train_df, test_df = train_test_split(
            df, 
            test_size=(1 - train_ratio), 
            random_state=42
        )

    # Save to CSV
    train_csv_path = os.path.join(root_dir, 'train_dataset.csv')
    test_csv_path = os.path.join(root_dir, 'test_dataset.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Saved train CSV to: {train_csv_path}")
    print(f"Saved test CSV to: {test_csv_path}")

#different distance test
def create_dataset_csv2(root_dir):
    # root_dir = r"E:\dataset\ultrasound_video_audio\DATA\dataset"
    target_dirs = ['ldx0', 'ldx1', 'ldx2', 'ldx3', 'ldx4', 'ldx5']
    label = 1

    all_experiments = []
    
    # Process only the specific target directory
    for user in target_dirs:
        user_path = os.path.join(root_dir, user)
    
        if not os.path.exists(user_path):
            print(f"Error: Directory {user_path} not found.")
            return

        # Find all .mp4 files
        video_files = glob.glob(os.path.join(user_path, "*_video.mp4"))
        
        for vid_path in video_files:
            base_name_with_path = vid_path.replace('_video.mp4', '')
            base_name = os.path.basename(base_name_with_path)
            
            audio_path = f"{base_name_with_path}_audio.wav"
            ultrasound_path = f"{base_name_with_path}_us.npy"
            video_feature_path = f"{base_name_with_path}_video.npy"
            
            if os.path.exists(audio_path) and os.path.exists(ultrasound_path) and os.path.exists(video_feature_path):
                sample_info = {
                    'user_name': user,
                    'label': label,
                    'video_path': vid_path,
                    'audio_path': audio_path,
                    'ultrasound_path': ultrasound_path,
                    'sample_name': base_name,
                    'video_feature_path': video_feature_path,
                    'TrueUser': 1
                }
                all_experiments.append(sample_info)
            else:
                print(f"Warning: Missing files for {vid_path}, skipping.")

        if not all_experiments:
            print("No valid samples found.")
            return

    # Convert to DataFrame
    df = pd.DataFrame(all_experiments)
    
    # Save to CSV
    # Defining it as test_dataset_ldxTest.csv as per likely usage for specific test set
    output_csv_path = os.path.join(root_dir, 'test_dataset_ldxTest.csv')
    
    df.to_csv(output_csv_path, index=False)
    
    print(f"Total samples found: {len(df)}")
    print(f"Saved CSV to: {output_csv_path}")


#构建负样本数据集，将上述得到的正样本的video_path，audio_path，ultrasound_path进行打乱重组，生成负样本数据集，三者不要保证相同的，然后TrueUser标记为0,保存为新的_fakeMismatch.csv文件。直接从上述的csv文件读取进行变换即可，负样本数量保证是正样本的5倍
def create_negative_samples_csv():
    root_dir = r"E:\dataset\ultrasound_video_audio\DATA\dataset"
    # input_csv_path = os.path.join(root_dir, 'test_dataset_ldxTest.csv')
    # output_csv_path = os.path.join(root_dir, 'test_dataset_ldxTest_fakeMismatch.csv')
    
    input_csv_path = os.path.join(root_dir, 'test_dataset.csv')
    output_csv_path = os.path.join(root_dir, 'test_dataset_fakeMismatch.csv')

    # input_csv_path = os.path.join(root_dir, 'train_dataset.csv')
    # output_csv_path = os.path.join(root_dir, 'train_dataset_fakeMismatch.csv')

    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV {input_csv_path} not found.")
        return

    df = pd.read_csv(input_csv_path)

    video_paths = df['video_path'].tolist()
    audio_paths = df['audio_path'].tolist()
    ultrasound_paths = df['ultrasound_path'].tolist()
    
    num_samples = len(df)
    negative_samples = []

    import random

    for _ in range(num_samples * 5):  # 5 times the number of positive samples
        vid_path = random.choice(video_paths)
        aud_path = random.choice(audio_paths)
        us_path = random.choice(ultrasound_paths)

        # Ensure they are not from the same sample
        while (vid_path.replace('_video.mp4', '') == aud_path.replace('_audio.wav', '') or
               vid_path.replace('_video.mp4', '') == us_path.replace('_us.npy', '') or
               aud_path.replace('_audio.wav', '') == us_path.replace('_us.npy', '')):
            vid_path = random.choice(video_paths)
            aud_path = random.choice(audio_paths)
            us_path = random.choice(ultrasound_paths)

        base_name_with_path = vid_path.replace('_video.mp4', '')
        base_name = os.path.basename(base_name_with_path)

        sample_info = {
            'user_name': 'mismatch',
            'label': -1,
            'video_path': vid_path,
            'audio_path': aud_path,
            'ultrasound_path': us_path,
            'sample_name': base_name,
            'video_feature_path': base_name_with_path + '_video.npy',
            'TrueUser': 0
        }
        negative_samples.append(sample_info)

    neg_df = pd.DataFrame(negative_samples)
    neg_df.to_csv(output_csv_path, index=False)

    print(f"Generated {len(neg_df)} negative samples.")
    print(f"Saved negative samples CSV to: {output_csv_path}")    




if __name__ == "__main__":
    # Note: Using raw string for Windows path to avoid escape character issues
    # dataset_root = r"E:\dataset\ultrasound_video_audio\DATA\dataset"
    dataset_root = "/root/autodl-tmp/UltraPrint_dataset/dataset_npy/"

    # create_dataset_csv(dataset_root, train_ratio=0.9)

    create_dataset_csv2()


    create_negative_samples_csv()