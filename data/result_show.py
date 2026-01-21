import os
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

SAVE_DIR = "./results/stage1"


def load_data(root_path, samples_per_user=50):
    data = []
    labels = []
    user_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    print(f"Found {len(user_folders)} users.")

    for user_idx, user_folder in enumerate(user_folders):
        user_path = os.path.join(root_path, user_folder)
        files = [f for f in os.listdir(user_path) if f.endswith('_midPool.npy')]
        
        # Select up to samples_per_user
        selected_files = files[:samples_per_user]
        
        for file_name in selected_files:
            file_path = os.path.join(user_path, file_name)
            try:
                sample = np.load(file_path)
                # Flatten the sample if it's multidimensional
                data.append(sample.flatten())
                labels.append(user_folder) # Use folder name as label
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return np.array(data), np.array(labels)

def plot_tsne_2d(X_embedded, labels, title="t-SNE 2D Visualization"):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=label, alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    OUTPUT_DIR = os.path.join(SAVE_DIR, "pc_others_tsne_2d.jpg")
    plt.savefig(OUTPUT_DIR)
    plt.show()
    plt.close()
    

def plot_tsne_3d(X_embedded, labels, title="t-SNE 3D Visualization"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)
        ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1], X_embedded[indices, 2], label=label, alpha=0.7)
    
    ax.set_title(title)
    ax.legend()
    OUTPUT_DIR = os.path.join(SAVE_DIR, "pc_others_tsne_3d.jpg")
    plt.savefig(OUTPUT_DIR)
    plt.show()
    plt.close()

def main():
    # Path configuration
    # Note: Python strings use backslashes as escape characters, so use raw strings (r"...") or forward slashes
    data_path = r"E:\dataset\ultrasound_video_audio\record\stage1_test_dataset"



    if not os.path.exists(data_path):
        print(f"Path not found: {data_path}")
        return

    print("Loading data...")
    X, y = load_data(data_path, samples_per_user=100)
    
    if len(X) == 0:
        print("No data loaded.")
        return
        
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # t-SNE 2D
    print("Running t-SNE (2D)...")
    tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)
    plot_tsne_2d(X_2d, y)

    # t-SNE 3D
    print("Running t-SNE (3D)...")
    tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
    X_3d = tsne_3d.fit_transform(X)
    plot_tsne_3d(X_3d, y)

if __name__ == "__main__":
    main()