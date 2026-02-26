import os
import random
import numpy as np

# Paths
base_anchor_path = '/root/autodl-tmp/Results/test_dataset'

#impact phone
base_target_path = '/root/autodl-tmp/Results/test_dataset_impact_phone/impact_phone'  # Assuming files are inside this subfolder based on exploration

#impact distance
# base_target_path = '/root/autodl-tmp/Results/test_dataset_impact_dis/impact_dis'  # Assuming files are inside this subfolder based on exploration

#impact distance
# base_target_path = '/root/autodl-tmp/Results/test_dataset_impact_angle/impact_angle'  # Assuming files are inside this subfolder based on exploration


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

# Load data helper (Modified for single file path usage mostly, or constructing paths)
def load_triplet_from_paths(u_path, v_path, fusion_path):
    if os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(fusion_path):
        u_vec = np.load(u_path).flatten()
        v_vec = np.load(v_path).flatten()
        fusion_vec = np.load(fusion_path).flatten()
        return u_vec, v_vec, fusion_vec
    return None, None, None

def load_anchor_triplet(user_path, base_filename):
    u_path = os.path.join(user_path, f"{base_filename}_u.npy")
    v_path = os.path.join(user_path, f"{base_filename}_v.npy")
    fusion_path = os.path.join(user_path, f"{base_filename}_fusion.npy")
    return load_triplet_from_paths(u_path, v_path, fusion_path)

# Build list of anchors
users_anchor = [d for d in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, d))]

# Output file
# impact_acc_file_path = os.path.join(output_dir, 'impact_accuracy_dis.txt')
impact_acc_file_path = os.path.join(output_dir, 'impact_accuracy_angle.txt')

impact_acc_file = open(impact_acc_file_path, 'w')
# Header: Impact_Type | Impact_Level | User | Accuracy | Count
impact_acc_file.write("Impact_Type\tImpact_Level\tUser\tAccuracy\tCorrect\tTotal\n")

target_bar = 0.3
all_anchors = [] 

# Step 0: Select Anchors (one random anchor per user from base_anchor_path)
print("Selecting Anchors...")
for user in users_anchor:
    user_dir = os.path.join(base_anchor_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    
    if not base_filenames:
        continue

    # Randomly select one anchor
    anchor_base = random.choice(base_filenames)
    anchor_u, anchor_v, anchor_fusion = load_anchor_triplet(user_dir, anchor_base)
    
    if anchor_u is None:
        # Try finding a valid one
        found_valid = False
        for alt_base in base_filenames:
            anchor_u, anchor_v, anchor_fusion = load_anchor_triplet(user_dir, alt_base)
            if anchor_u is not None:
                anchor_base = alt_base
                found_valid = True
                break
        if not found_valid:
            continue

    all_anchors.append({
        'user': user,
        'u': anchor_u,
        'v': anchor_v,
        'fusion': anchor_fusion
    })
print(f"Loaded {len(all_anchors)} Anchors.")


# Step 1: Process Impact Samples
print("Processing Impact Samples...")

# Data structure to hold results:
# results[impact_type][impact_level][user] = {'correct': 0, 'total': 0}
results = {}

if os.path.exists(base_target_path):
    # Iterate over all files in the impact directory
    files = os.listdir(base_target_path)
    
    # Filter for _v.npy to identify unique samples (we need u, v, fusion)
    v_files = [f for f in files if f.endswith('_v.npy')]
    
    for v_file in v_files:
        # Parse filename
        # Pattern: Impact_SpecificImpact_ImpactLevel_UserName_seg_SampleIndex_video_v.npy
        # Example: impact_phone_S22Ultra_ldx_seg_0002_video_v.npy
        
        parts = v_file.split('_')
        # Check if structure matches expected minimum length
        # impact(0), phone(1), S22Ultra(2), ldx(3), seg(4), 0002(5), video(6), v.npy(7) -> 8 parts usually
        
        if len(parts) < 8:
            continue
            
        impact_prefix = parts[0] # "impact"
        specific_impact = parts[1] # "phone" / "angle" etc.
        impact_level = parts[2]    # "S22Ultra" / "down"
        user_name = parts[3]       # "ldx"
        
        # Verify user exists in anchors?
        # The prompt says "first ensure both have same user name".
        # If user is not in our anchor set, we can't verify "Is this user X?". 
        # But maybe we should check if we HAVE an anchor for this user.
        if user_name not in users_anchor:
            continue

        base_name = v_file.replace('_v.npy', '')
        
        # Load sample triplet
        u_path = os.path.join(base_target_path, f"{base_name}_u.npy")
        v_path = os.path.join(base_target_path, f"{base_name}_v.npy")
        fusion_path = os.path.join(base_target_path, f"{base_name}_fusion.npy")
        
        curr_u, curr_v, curr_fusion = load_triplet_from_paths(u_path, v_path, fusion_path)
        
        if curr_u is None:
            continue

        # Logic to match against anchors
        predicted_user = None
        
        # 1. Find anchors with V similarity > threshold
        candidate_anchors = []
        for anchor_obj in all_anchors:
            sim_v = cosine_similarity(anchor_obj['v'], curr_v)
            if sim_v > target_bar:
                candidate_anchors.append({
                    'user': anchor_obj['user'],
                    'sim_v': sim_v,
                    'u': anchor_obj['u'],
                    'fusion': anchor_obj['fusion']
                })
        
        # 2. If candidates exist, pick highest V similarity
        if candidate_anchors:
            candidate_anchors.sort(key=lambda x: x['sim_v'], reverse=True)
            best_anchor = candidate_anchors[0]
            
            print(f"Sample {base_name}: Found candidate anchor {best_anchor['user']} with V sim {best_anchor['sim_v']:.4f}")
            
            # 3. Check U and Fusion similarity
            sim_u = cosine_similarity(best_anchor['u'], curr_u)
            sim_fusion = cosine_similarity(best_anchor['fusion'], curr_fusion)
            
            print(f"Sample {base_name}: U sim {sim_u:.4f}, Fusion sim {sim_fusion:.4f}")
            
            if sim_u > target_bar and sim_fusion > target_bar:
            # if sim_fusion > target_bar:
                predicted_user = best_anchor['user']
            else:
                predicted_user = None # Rejected
        else:
            predicted_user = None
        
        # Record Result
        if specific_impact not in results:
            results[specific_impact] = {}
        if impact_level not in results[specific_impact]:
            results[specific_impact][impact_level] = {}
        if user_name not in results[specific_impact][impact_level]:
            results[specific_impact][impact_level][user_name] = {'correct': 0, 'total': 0}
            
        if predicted_user == user_name:
            results[specific_impact][impact_level][user_name]['correct'] += 1
        results[specific_impact][impact_level][user_name]['total'] += 1

else:
    print(f"Error: Impact data path {base_target_path} does not exist.")

# Write results to file
# Sort keys for consistent output
sorted_impacts = sorted(results.keys())
for impact in sorted_impacts:
    sorted_levels = sorted(results[impact].keys())
    for level in sorted_levels:
        sorted_users = sorted(results[impact][level].keys())
        for user in sorted_users:
            stats = results[impact][level][user]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            impact_acc_file.write(f"{impact}\t{level}\t{user}\t{acc:.4f}\t{stats['correct']}\t{stats['total']}\n")
            print(f"Processed {impact}/{level}/{user}: Acc={acc:.4f} ({stats['correct']}/{stats['total']})")

impact_acc_file.close()
print(f"Results saved to {impact_acc_file_path}")
