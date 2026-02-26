import os
import random
import numpy as np

base_anchor_path = '/root/autodl-tmp/Results/test_dataset'
base_target_path = '/root/autodl-tmp/Results/test_dataset_fakeMismatch'

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
def load_fused_vector(user_path, base_filename):
    # Determine possible filenames. 
    # cal_result4.py used _fusion.npy and _v.npy
    # User prompt mentioned _fusion.py but likely meant _fusion.npy as it is vector data.
    mid_path = os.path.join(user_path, f"{base_filename}_fusion.npy")
    vid_path = os.path.join(user_path, f"{base_filename}_v.npy")
    
    if os.path.exists(mid_path) and os.path.exists(vid_path):
        mid_vec = np.load(mid_path).flatten()
        vid_vec = np.load(vid_path).flatten()
        
        # Fuse vectors by addition
        if mid_vec.shape == vid_vec.shape:
            return mid_vec + vid_vec
        else:
            # Handle shape mismatch if any, though unlikely based on context
            # Assuming broadcasting or error if shapes differ essentially
            # For now try direct addition
            try:
                return mid_vec + vid_vec
            except ValueError:
                print(f"Shape mismatch for {base_filename}: {mid_vec.shape} vs {vid_vec.shape}")
                return None
            
    return None

# Main processing
users = [d for d in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, d))]
anchor_file = open(os.path.join(output_dir, 'anchors_cal5.txt'), 'w')
pos_acc_file = open(os.path.join(output_dir, 'positive_accuracy_cal5.txt'), 'w')
neg_acc_file = open(os.path.join(output_dir, 'negative_accuracy_cal5.txt'), 'w')

total_pos_correct = 0
total_pos_count = 0
total_neg_correct = 0
total_neg_count = 0
target_bar = 0.5

all_anchors = [] # List of (user_name, fused_vec)

# Step 0: Select Anchors for all users first
print("Selecting Anchors...")
for user in users:
    user_dir = os.path.join(base_anchor_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue

    # Randomly select one anchor
    anchor_base = random.choice(base_filenames)
    anchor_vec = load_fused_vector(user_dir, anchor_base)
    
    if anchor_vec is None:
        continue

    # Save anchor info
    anchor_path_str = f"{os.path.join(user_dir, anchor_base)}"
    anchor_file.write(f"{user}\t{anchor_path_str}\n")
    all_anchors.append((user, anchor_vec))

# Step 1: Process Positive Samples (Classification)
print("Processing Positive Samples...")
for user in users:
    user_dir = os.path.join(base_anchor_path, user)
    
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue

    user_correct = 0
    user_total = 0
    
    for base in base_filenames:
        curr_vec = load_fused_vector(user_dir, base)
        if curr_vec is None: continue
        
        # Compare with ALL anchors
        best_sim_score = -1.0
        best_anchor_user = None
        matched_any = False
        
        for anchor_user, anchor_vec in all_anchors:
            sim = cosine_similarity(anchor_vec, curr_vec)
            
            # print(f"sample name: {base}, anchor_user: {anchor_user}, sim: {sim}")
            
            # Criteria: Similarity > target_bar
            if sim > target_bar:
                matched_any = True
                
                if sim > best_sim_score:
                    best_sim_score = sim
                    best_anchor_user = anchor_user
        
        # Determine correctness
        if matched_any:
            if best_anchor_user == user:
                user_correct += 1
        # If not matched_any, it is incorrect (miss)
        
        user_total += 1
    
    acc = user_correct / user_total if user_total > 0 else 0
    pos_acc_file.write(f"{user}\t{acc:.4f}\n")
    
    total_pos_correct += user_correct
    total_pos_count += user_total

overall_pos_acc = total_pos_correct / total_pos_count if total_pos_count > 0 else 0
pos_acc_file.write(f"OVERALL\t{overall_pos_acc:.4f}\n")
print(f"Overall Positive Accuracy: {overall_pos_acc:.4f}")

# Step 2: Process Negative Samples (Mismatch/Rejection)
print("Processing Negative Samples...")
for user in users:
    # Target path usually contains same user folders but with negative samples inside
    target_user_dir = os.path.join(base_target_path, user)
    if not os.path.exists(target_user_dir):
        continue
        
    files = [f for f in os.listdir(target_user_dir) if f.endswith('_v.npy')] # Check target dir for files
    # Note: user_dir was used in cal_result4.py copy-paste error likely in Step 2 loop definition of files
    # Correcting to check target_user_dir
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue
        
    user_neg_correct = 0
    user_neg_total = 0
    
    for base in base_filenames:
        curr_vec = load_fused_vector(target_user_dir, base)
        if curr_vec is None: continue
        
        # Check against ALL anchors
        matches_any_anchor = False
        
        for anchor_user, anchor_vec in all_anchors:
            sim = cosine_similarity(anchor_vec, curr_vec)
            
            if sim > target_bar:
                matches_any_anchor = True
                break
        
        # "If current sample and all anchors' similarity < target_bar then considered correct"
        # "Otherwise if it has similarity > target_bar with any anchor, considered incorrect"
        if not matches_any_anchor:
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
