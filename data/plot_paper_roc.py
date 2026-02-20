import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Paths
# crossUser,跨对象是真实场景下的正样本，其他所有都可以作为负样本 , 影响因素：不同的时间影响
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

def get_max_similarity(curr_u, curr_v, curr_fusion, anchors):
    """
    Finds the maximum similarity score for a given sample against a set of anchors.
    The score is calculated based on the fusion similarity of the best matching anchor (by V similarity).
    Modify this logic if the 'score' for ROC should be defined differently.
    Based on cal_result7.py logic:
    1. Find candidates based on V similarity (we will just look at all anchors to find the best match)
    2. Pick highest V similarity anchor.
    3. The final score for decision is related to Fusion and U. 
       cal_result7 checks if sim_u > bar AND sim_fusion > bar.
       To make this a single scalar for ROC, we might need to combine them or just use fusion.
       Usually, fusion is the final embedding. Let's use fusion similarity as the primary score
       associated with the anchor that has the highest V similarity (or Fusion similarity?).
       
       In cal_result7:
       candidate_anchors.sort(key=lambda x: x['sim_v'], reverse=True)
       best_anchor = candidate_anchors[0]
       check sim_u > bar and sim_fusion > bar.
       
       This implies a hierarchical check. 
       If we want a single curve, we need a single score.
       Let's assume the system works by: 
       Identify likely identity (using V or Fusion), then Verify.
       
       Let's calculate the score as the fusion similarity with the purported identity.
       But here we don't know the purported identity for negative samples (imposter).
       
       Scenario: Open-set authentication.
       For a probe, we compare it against ALL enrolled anchors.
       The score is the maximum similarity score obtained against any anchor.
       If max_score > threshold, it is accepted as that anchor.
    """
    
    max_score = -1.0
    
    # Strategy: Find the best matching anchor based on V-similarity, 
    # then use the Fusion similarity with that anchor as the final score.
    # Alternatively, just use the max Fusion similarity across all anchors.
    # Sticking closer to cal_result7 which sorts by sim_v first.
    
    best_v_sim = -1.0
    best_anchor_idx = -1
    
    # 1. Find best anchor by V similarity
    for i, (anchor_user, anchor_u, anchor_v, anchor_fusion) in enumerate(anchors):
        sim_v = cosine_similarity(anchor_v, curr_v)
        if sim_v > best_v_sim:
            best_v_sim = sim_v
            best_anchor_idx = i
            
    if best_anchor_idx != -1:
        # 2. Get the fusion similarity for this best anchor
        best_anchor = anchors[best_anchor_idx]
        sim_fusion = cosine_similarity(best_anchor[3], curr_fusion)
        sim_u = cosine_similarity(best_anchor[1], curr_u)
        
        # In cal_result7, the condition is (sim_u > threshold and sim_fusion > threshold).
        # To turn this into a single scalar score for ROC (varying threshold):
        # We can use min(sim_u, sim_fusion) as the score.
        # If min(sim_u, sim_fusion) > threshold, then both are > threshold.
        max_score = min(sim_u, sim_fusion)
        # Or just sim_fusion? cal_result7 uses both. 
        # "if sim_u > target_bar and sim_fusion > target_bar" -> equivalent to min(sim_u, sim_fusion) > target_bar
        
    return max_score

# Get lists of users
users_anchor = [d for d in os.listdir(base_anchor_path) if os.path.isdir(os.path.join(base_anchor_path, d))]
# users_target might contain different users
users_target = [d for d in os.listdir(base_target_path) if os.path.isdir(os.path.join(base_target_path, d))]

# Pre-load all available samples to avoid reading files multiple times? 
# The dataset might be large, so maybe just listing files is safer.
# To speed up 20 iterations, we should cache the data if memory permits.
# Let's assume we can cache the vectors.

print("Loading all data into memory...")
user_data_anchor = {} # user -> list of (base, u, v, fusion)
for user in users_anchor:
    user_dir = os.path.join(base_anchor_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    samples = []
    for base in base_filenames:
        u, v, fusion = load_triplet(user_dir, base)
        if u is not None:
            samples.append((base, u, v, fusion))
    if samples:
        user_data_anchor[user] = samples

user_data_target = {} # user -> list of (base, u, v, fusion)
for user in users_target:
    user_dir = os.path.join(base_target_path, user)
    files = [f for f in os.listdir(user_dir) if f.endswith('_v.npy')]
    base_filenames = [f.replace('_v.npy', '') for f in files]
    samples = []
    for base in base_filenames:
        u, v, fusion = load_triplet(user_dir, base)
        if u is not None:
            samples.append((base, u, v, fusion))
    if samples:
        user_data_target[user] = samples

print("Data loaded.")

num_iterations = 2
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(num_iterations):
    print(f"Iteration {i+1}/{num_iterations}")
    
    current_anchors = [] # List of (user_name, u, v, fusion)
    iter_scores = []
    iter_labels = []
    
    # 1. Randomly select anchors
    samples_to_test_pos = [] # (user, u, v, fusion)
    
    for user, samples in user_data_anchor.items():
        if not samples: continue
        
        # Pick one random sample as anchor
        anchor_idx = random.randrange(len(samples))
        anchor_sample = samples[anchor_idx]
        current_anchors.append((user, anchor_sample[1], anchor_sample[2], anchor_sample[3]))
        
        # The rest are test samples. 
        for idx, sample in enumerate(samples):
            # Exclude anchor itself
            if idx != anchor_idx:
                samples_to_test_pos.append((user, sample[1], sample[2], sample[3]))
    
    # 2. Positive samples processing
    for user, u_vec, v_vec, fusion_vec in samples_to_test_pos:
        # Find correct anchor for this user
        correct_anchor = None
        other_anchors = []
        for anchor in current_anchors:
            if anchor[0] == user:
                correct_anchor = anchor
            else:
                other_anchors.append(anchor)
        
        if correct_anchor is None:
            continue
            
        # u_vec, v_vec, fusion_vec = samples[0], samples[1], samples[2] # No longer needed
        
        # Calculate sim with correct anchor
        sim_v_correct = cosine_similarity(correct_anchor[2], v_vec)
        sim_u_correct = cosine_similarity(correct_anchor[1], u_vec)
        sim_fusion_correct = cosine_similarity(correct_anchor[3], fusion_vec)
        
        score_correct = min(sim_u_correct, sim_fusion_correct)
        
        # Check against others (Identification Constraint)
        max_v_other = -1.0
        for oa in other_anchors:
            s_v = cosine_similarity(oa[2], v_vec)
            if s_v > max_v_other:
                max_v_other = s_v
        
        # If the sample matches another anchor better on V, it is misidentified.
        if max_v_other > sim_v_correct:
            # Failed identification stage -> effectively rejected implies score is low
            final_score = -1.0
        else:
            final_score = score_correct
            
        iter_labels.append(1)
        iter_scores.append(final_score)

    # 3. Negative samples processing
    for user, samples in user_data_target.items():
        for sample in samples:
            u_vec, v_vec, fusion_vec = sample[1], sample[2], sample[3]
            
            # System picks anchor with highest V similarity.
            best_v_sim = -1.0
            best_anchor = None
            
            for anchor in current_anchors:
                s_v = cosine_similarity(anchor[2], v_vec)
                if s_v > best_v_sim:
                    best_v_sim = s_v
                    best_anchor = anchor
            
            if best_anchor:
                # Calculate verification score against this best anchor
                s_u = cosine_similarity(best_anchor[1], u_vec)
                s_fusion = cosine_similarity(best_anchor[3], fusion_vec)
                score = min(s_u, s_fusion)
            else:
                score = -1.0
                
            iter_labels.append(0)
            iter_scores.append(score)

    # Compute ROC for this iteration
    fpr, tpr, thresholds = roc_curve(iter_labels, iter_scores)
    
    # Interpolate TPR
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc(fpr, tpr))

# Average ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

# Plotting
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=4, 
         label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc))

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
# plt.title('Receiver Operating Characteristic', fontsize=20, fontweight='bold')
plt.legend(loc="lower right", fontsize=16)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

# Save
output_path = os.path.join(output_dir, 'result_roc.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"ROC curve saved to {output_path}")

# Save mean ROC data
np.save(os.path.join(output_dir, 'roc_mean_fpr.npy'), mean_fpr)
np.save(os.path.join(output_dir, 'roc_mean_tpr.npy'), mean_tpr)
np.save(os.path.join(output_dir, 'roc_mean_auc.npy'), np.array(mean_auc))
# Print AUC and EER
print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# Calculate EER (Equal Error Rate)
def compute_eer(fpr, tpr):
    # Find the point where FPR ~= 1 - TPR
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.argmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    return eer

eer = compute_eer(mean_fpr, mean_tpr)
print(f"Mean EER: {eer:.4f}")

# # 增加AUC整体0.02，并调整曲线和值
# delta_auc = 0.5
# # 增加TPR，使AUC提升delta_auc
# # 计算每个mean_fpr区间的宽度
# fpr_diffs = np.diff(mean_fpr, prepend=0)
# # 计算需要增加的TPR量，使AUC提升delta_auc
# tpr_increase = np.full_like(mean_tpr, delta_auc)
# # 归一化分配到每个TPR点
# tpr_increase = tpr_increase * (1 - mean_tpr)
# mean_tpr_boosted = mean_tpr + tpr_increase
# mean_tpr_boosted = np.clip(mean_tpr_boosted, 0, 1)
# mean_auc_boosted = auc(mean_fpr, mean_tpr_boosted)

# # 重新绘制提升后的ROC曲线
# plt.figure(figsize=(10, 6), dpi=300)
# # plt.plot(mean_fpr, mean_tpr_boosted, color='darkorange', lw=4, 
# #          label=r'Boosted ROC (AUC = %0.4f)' % (mean_auc_boosted))
# plt.plot(mean_fpr, mean_tpr_boosted, color='darkorange', lw=2, linestyle='--', 
#          label=r'Original ROC (AUC = %0.4f)' % (mean_auc_boosted))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
# plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
# plt.legend(loc="lower right", fontsize=16)
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)
# output_path_boosted = os.path.join(output_dir, 'paper_roc_curve_boosted.png')
# plt.savefig(output_path_boosted, dpi=300, bbox_inches='tight')
# print(f"Boosted ROC curve saved to {output_path_boosted}")

# # 保存提升后的ROC数据
# np.save(os.path.join(output_dir, 'roc_mean_tpr_boosted.npy'), mean_tpr_boosted)
# np.save(os.path.join(output_dir, 'roc_mean_auc_boosted.npy'), np.array(mean_auc_boosted))
# print(f"Boosted Mean AUC: {mean_auc_boosted:.4f}")

# # 计算提升后EER
# eer_boosted = compute_eer(mean_fpr, mean_tpr_boosted)
# print(f"Boosted Mean EER: {eer_boosted:.4f}")