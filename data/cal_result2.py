import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_npy(path):
    """Safely load a .npy file and flatten it for similarity calculation."""
    try:
        data = np.load(path)
        return data.flatten().reshape(1, -1)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    # Paths
    # base_anchor_path = '/root/autodl-tmp/Results/test_dataset'
    # base_target_path = '/root/autodl-tmp/Results/test_dataset_fakeMismatch'
    
    base_anchor_path = '/root/autodl-tmp/Results/stage1_test_dataset'
    base_target_path = '/root/autodl-tmp/Results/stage1_test_dataset_fakeMismatch'
    
    output_dir = '/root/UltraPrint/data/show_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    anchor_info_file = os.path.join(output_dir, 'anchor_sample.txt')
    result_file = os.path.join(output_dir, 'similarity_result.txt')

    # --- Step 1: Select Anchor Samples ---
    print("Selecting anchor samples...")
    anchor_samples = []  # List of dicts: {'id': int, 'path': str, 'user': str, 'data': numpy array}
    
    user_folders = [f for f in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, f))]
    user_folders.sort() # Ensure consistent ID assignment
    
    with open(anchor_info_file, 'w') as f_anchor:
        for idx, user_folder in enumerate(user_folders):
            user_path = os.path.join(base_anchor_path, user_folder)
            files = [f for f in os.listdir(user_path) if f.endswith('_fusion.npy')]
            
            if files:
                selected_file = random.choice(files)
                full_path = os.path.join(user_path, selected_file)
                
                # Load data immediately to cache it for comparisons
                data = load_npy(full_path)
                
                if data is not None:
                    anchor_info = {
                        'id': idx,
                        'user': user_folder,
                        'path': full_path,
                        'name': selected_file,
                        'data': data
                    }
                    anchor_samples.append(anchor_info)
                    f_anchor.write(f"ID: {idx}, User: {user_folder}, Name: {selected_file}, Path: {full_path}\n")

    print(f"Recorded {len(anchor_samples)} anchor samples to {anchor_info_file}")






    # --- Step 2: Compare Target Samples ---
    print("Comparing target samples...")
    total_samples = 0
    correct_matches = 0 # Similarity > 0.5
    similarity_bar = 0.5
    
    target_users = [f for f in os.listdir(base_target_path) if os.path.isdir(os.path.join(base_target_path, f))]
    
    # Pre-stack anchor data for faster matrix multiplication
    if not anchor_samples:
        print("No anchor samples found. Exiting.")
        return

    anchor_matrix = np.vstack([x['data'] for x in anchor_samples])
    anchor_ids = [x['id'] for x in anchor_samples]

    with open(result_file, 'w') as f_result:
        for user_folder in target_users:
            user_path = os.path.join(base_target_path, user_folder)
            files = [f for f in os.listdir(user_path) if f.endswith('_fusion.npy')]
            
            for file_name in files:
                target_path = os.path.join(user_path, file_name)
                target_data = load_npy(target_path)
                
                if target_data is not None:
                    total_samples += 1
                    
                    # Calculate similarity against all anchors at once
                    # shape: (1, n_features) * (n_anchors, n_features).T -> (1, n_anchors)
                    similarities = cosine_similarity(target_data, anchor_matrix)[0]
                    
                    # Find max similarity and corresponding index
                    max_sim_idx = np.argmax(similarities)
                    max_sim_val = similarities[max_sim_idx]
                    best_match_id = anchor_ids[max_sim_idx]
                    
                    if max_sim_val > similarity_bar:
                        correct_matches += 1
                    
                    log_line = (f"Sample: {file_name}, Path: {target_path}, "
                                f"Best_Anchor_ID: {best_match_id}, Similarity: {max_sim_val:.6f}")
                    f_result.write(log_line + "\n")

    # --- Step 3: Calculate Statistics ---
    accuracy = (correct_matches / total_samples) if total_samples > 0 else 0.0
    
    stats_msg = f"\nTotal Samples: {total_samples}, Matches > {similarity_bar}: {correct_matches}, Accuracy: {accuracy:.4f}"
    print(stats_msg)
    
    # Append stats to results file
    with open(result_file, 'a') as f_result:
        f_result.write(stats_msg)

if __name__ == "__main__":
    main()