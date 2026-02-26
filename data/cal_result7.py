import os
import random
import numpy as np

# Paths

#crossUser,跨对象是真实场景下的正样本，其他所有都可以作为负样本 , 影响因素：不同的时间影响
base_anchor_path = '/root/autodl-tmp/Results/test_dataset'
base_target_path = '/root/autodl-tmp/Results/test_dataset_fakeMismatch'

#mismatch，不匹配是原始的样本非跨对象划分
# base_anchor_path = '/root/autodl-tmp/Results/test_dataset'
# base_target_path = '/root/autodl-tmp/Results/test_dataset_fakeMismatch'

#cf,重放的负样本和模型已经存在的用户进行比较，这里重放的是ldx和lgd，这两者参与了训练，所以和test_dataset比较
# base_anchor_path = '/root/autodl-tmp/Results/test_dataset'
# base_target_path = '/root/autodl-tmp/Results/test_dataset_fake_cf'

#换脸的负样本，对注册的用户进行攻击，那么注册的用户作为正样本，换脸则使用其他用户的脸换成注册人的脸，然后进行比较
# base_anchor_path = '/root/autodl-tmp/Results/test_dataset_crossUser'
# base_target_path = '/root/autodl-tmp/Results/test_dataset_fake_hl'

#其他比较的影响因素：距离，角度，手机
# base_anchor_path = '/root/autodl-tmp/Results/test_dataset'

# base_target_path = '/root/autodl-tmp/Results/test_dataset_fake_dis'
# base_target_path = '/root/autodl-tmp/Results/test_dataset_fake_angle'
# base_target_path = '/root/autodl-tmp/Results/test_dataset_fake_phone'




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
def load_triplet(user_path, base_filename):
    u_path = os.path.join(user_path, f"{base_filename}_u.npy")
    v_path = os.path.join(user_path, f"{base_filename}_v.npy")
    fusion_path = os.path.join(user_path, f"{base_filename}_fusion.npy")
    
    if os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(fusion_path):
        u_vec = np.load(u_path).flatten()
        v_vec = np.load(v_path).flatten()
        fusion_vec = np.load(fusion_path).flatten()
        return u_vec, v_vec, fusion_vec
    return None, None, None

# Get lists of users
users_anchor = [d for d in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, d))]
# For negative tests, we look at ALL users in the target path, regardless of whether they exist in anchor path
users_target = [d for d in os.listdir(base_target_path) if os.path.isdir(os.path.join(base_target_path, d))]

anchor_file = open(os.path.join(output_dir, 'anchors_cal7.txt'), 'w')
pos_acc_file = open(os.path.join(output_dir, 'positive_accuracy_cal7.txt'), 'w')
neg_acc_file = open(os.path.join(output_dir, 'negative_accuracy_cal7.txt'), 'w')

total_pos_correct = 0
total_pos_count = 0
total_neg_correct = 0
total_neg_count = 0
target_bar = 0.3

all_anchors = [] # List of (user_name, u, v, fusion)

# Step 0: Select Anchors (from base_anchor_path)
print("Selecting Anchors...")
for user in users_anchor:
    user_dir = os.path.join(base_anchor_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue

    # Randomly select one anchor
    anchor_base = random.choice(base_filenames)
    anchor_u, anchor_v, anchor_fusion = load_triplet(user_dir, anchor_base)
    
    if anchor_u is None:
        # Try to find another anchor if this one is incomplete
        found_valid = False
        for alt_base in base_filenames:
            anchor_u, anchor_v, anchor_fusion = load_triplet(user_dir, alt_base)
            if anchor_u is not None:
                anchor_base = alt_base
                found_valid = True
                break
        if not found_valid:
            continue

    # Save anchor info
    anchor_path_str = f"{os.path.join(user_dir, anchor_base)}"
    anchor_file.write(f"{user}\t{anchor_path_str}\n")
    all_anchors.append((user, anchor_u, anchor_v, anchor_fusion))

# Step 1: Process Positive Samples (Classification)
# Samples come from base_anchor_path (same as anchors)
print("Processing Positive Samples...")
for user in users_anchor:
    user_dir = os.path.join(base_anchor_path, user)
    
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue

    user_correct = 0
    user_total = 0
    
    for base in base_filenames:
        curr_u, curr_v, curr_fusion = load_triplet(user_dir, base)
        if curr_u is None: continue
        
        # 1. Find anchors with V similarity > threshold
        candidate_anchors = []
        for anchor_user, anchor_u, anchor_v, anchor_fusion in all_anchors:
            sim_v = cosine_similarity(anchor_v, curr_v)
            if sim_v > target_bar:
                candidate_anchors.append({
                    'user': anchor_user,
                    'sim_v': sim_v,
                    'u': anchor_u,
                    'fusion': anchor_fusion
                })
        
        predicted_user = None
        
        # 2. If candidates exist, pick highest V similarity
        if candidate_anchors:
            candidate_anchors.sort(key=lambda x: x['sim_v'], reverse=True)
            best_anchor = candidate_anchors[0]
            
            # 3. Check U and Fusion similarity
            sim_u = cosine_similarity(best_anchor['u'], curr_u)
            sim_fusion = cosine_similarity(best_anchor['fusion'], curr_fusion)
            
            if sim_u > target_bar and sim_fusion > target_bar:
                predicted_user = best_anchor['user']
            else:
                predicted_user = None # Rejected
        else:
            predicted_user = None
        
        # Determine correctness (Positive Sample: must match user)
        if predicted_user == user:
            user_correct += 1
        
        user_total += 1
    
    acc = user_correct / user_total if user_total > 0 else 0
    pos_acc_file.write(f"{user}\t{acc:.4f}\n")
    
    total_pos_correct += user_correct
    total_pos_count += user_total

overall_pos_acc = total_pos_correct / total_pos_count if total_pos_count > 0 else 0
pos_acc_file.write(f"OVERALL\t{overall_pos_acc:.4f}\n")
print(f"Overall Positive Accuracy: {overall_pos_acc:.4f}")

# Step 2: Process Negative Samples (Mismatch/Rejection)
# Samples come from base_target_path (different folder structure, potentially different users)
print("Processing Negative Samples...")
for user in users_target:
    target_user_dir = os.path.join(base_target_path, user)
    # No checks for if user exists in anchors -- we treat all in target path as negatives
        
    files = [f for f in os.listdir(target_user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue
        
    user_neg_correct = 0
    user_neg_total = 0
    
    for base in base_filenames:
        curr_u, curr_v, curr_fusion = load_triplet(target_user_dir, base)
        if curr_u is None: continue
        
        # 1. Find anchors with V similarity > threshold
        candidate_anchors = []
        for anchor_user, anchor_u, anchor_v, anchor_fusion in all_anchors:
            sim_v = cosine_similarity(anchor_v, curr_v)
            if sim_v > target_bar:
                candidate_anchors.append({
                    'user': anchor_user,
                    'sim_v': sim_v,
                    'u': anchor_u,
                    'fusion': anchor_fusion
                })
        
        is_authenticated = False
        
        # 2. If candidates exist, pick highest V similarity
        if candidate_anchors:
            candidate_anchors.sort(key=lambda x: x['sim_v'], reverse=True)
            best_anchor = candidate_anchors[0]
            print(f"Sample {base}: Found candidate anchor {best_anchor['user']} with V sim {best_anchor['sim_v']:.4f}")
            # 3. Check U and Fusion similarity
            sim_u = cosine_similarity(best_anchor['u'], curr_u)
            sim_fusion = cosine_similarity(best_anchor['fusion'], curr_fusion)
            print(f"Sample {base}: U sim {sim_u:.4f}, Fusion sim {sim_fusion:.4f}")
            
            if sim_u > target_bar and sim_fusion > target_bar:
            # if sim_fusion > target_bar:
                is_authenticated = True
                # Predicted user would be best_anchor['user']
        
        # Determine correctness (Negative Sample)
        # Correct if NOT authenticated (Rejected)
        if not is_authenticated:
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
