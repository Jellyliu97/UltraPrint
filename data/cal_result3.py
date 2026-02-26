import os
import random
import numpy as np

base_anchor_path = '/root/autodl-tmp/Results/stage1_test_dataset'
base_target_path = '/root/autodl-tmp/Results/stage1_test_dataset_fakeMismatch'

output_dir = '/root/UltraPrint/data/show_results'

# Create results directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# Load data helper
def load_pair(user_path, base_filename):
    mid_path = os.path.join(user_path, f"{base_filename}_midPool.npy")
    vid_path = os.path.join(user_path, f"{base_filename}_meanVideoFeature.npy")
    
    if os.path.exists(mid_path) and os.path.exists(vid_path):
        mid_vec = np.load(mid_path).flatten()
        vid_vec = np.load(vid_path).flatten()
        return mid_vec, vid_vec
    return None, None

# Main processing
users = [d for d in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, d))]
anchor_file = open(os.path.join(output_dir, 'anchors.txt'), 'w')
pos_acc_file = open(os.path.join(output_dir, 'positive_accuracy.txt'), 'w')
neg_acc_file = open(os.path.join(output_dir, 'negative_accuracy.txt'), 'w')

total_pos_correct = 0
total_pos_count = 0
total_neg_correct = 0
total_neg_count = 0
target_bar = 0.1

anchors = {} # Map user -> (mid_vec, vid_vec)

# Step 1: Select Anchors and Process Positive Samples
print("Processing Positive Samples...")
for user in users:
    user_dir = os.path.join(base_anchor_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_midPool.npy')]
    base_filenames = [f.replace('_midPool.npy', '') for f in files]
    
    if not base_filenames:
        continue

    # Randomly select one anchor
    anchor_base = random.choice(base_filenames)
    anchor_mid, anchor_vid = load_pair(user_dir, anchor_base)
    
    if anchor_mid is None:
        continue

    # Save anchor info
    anchor_path_str = f"{os.path.join(user_dir, anchor_base)}"
    anchor_file.write(f"{user}\t{anchor_path_str}\n")
    anchors[user] = (anchor_mid, anchor_vid)

    # Calculate Positive Accuracy for this user
    user_correct = 0
    user_total = 0
    
    for base in base_filenames:
        curr_mid, curr_vid = load_pair(user_dir, base)
        if curr_mid is None: continue
        
        sim_mid = cosine_similarity(anchor_mid, curr_mid)
        sim_vid = cosine_similarity(anchor_vid, curr_vid)
        
        # Positive criteria: both similarities > target_bar
        if sim_mid > target_bar and sim_vid > target_bar:
            user_correct += 1
        user_total += 1
    
    acc = user_correct / user_total if user_total > 0 else 0
    pos_acc_file.write(f"{user}\t{acc:.4f}\n")
    
    total_pos_correct += user_correct
    total_pos_count += user_total

overall_pos_acc = total_pos_correct / total_pos_count if total_pos_count > 0 else 0
pos_acc_file.write(f"OVERALL\t{overall_pos_acc:.4f}\n")
print(f"Overall Positive Accuracy: {overall_pos_acc:.4f}")

# Step 2: Process Negative Samples (Mismatch)
print("Processing Negative Samples...")
for user in users:
    if user not in anchors:
        continue
        
    anchor_mid, anchor_vid = anchors[user]
    
    # Target path usually contains same user folders but with negative samples inside
    target_user_dir = os.path.join(base_target_path, user)
    if not os.path.exists(target_user_dir):
        continue
        
    files = [f for f in os.listdir(target_user_dir) if f.endswith('_midPool.npy')]
    base_filenames = [f.replace('_midPool.npy', '') for f in files]
    
    user_neg_correct = 0
    user_neg_total = 0
    
    for base in base_filenames:
        curr_mid, curr_vid = load_pair(target_user_dir, base)
        if curr_mid is None: continue
        
        sim_mid = cosine_similarity(anchor_mid, curr_mid)
        sim_vid = cosine_similarity(anchor_vid, curr_vid)
        
        # Negative criteria: both similarities < target_bar
        if sim_mid < target_bar and sim_vid < target_bar:
            user_neg_correct += 1
        user_neg_total += 1
        
    acc = user_neg_correct / user_neg_total if user_neg_total > 0 else 0
    neg_acc_file.write(f"{user}\t{acc:.4f}\n")
    
    total_neg_correct += user_neg_correct
    total_neg_count += user_neg_total

overall_neg_acc = total_neg_correct / total_neg_count if total_neg_count > 0 else 0
neg_acc_file.write(f"OVERALL\t{overall_neg_acc:.4f}\n")
print(f"Overall Negative Accuracy: {overall_neg_acc:.4f}")

anchor_file.close()
pos_acc_file.close()
neg_acc_file.close()


