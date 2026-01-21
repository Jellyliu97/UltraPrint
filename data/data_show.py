import os
import numpy as np
from sklearn.manifold import TSNE
import glob

import matplotlib.pyplot as plt

def load_and_sample_data(root_path, samples_per_user=100):
    """
    Loads .npy files from user directories, samples data, and prepares labels.
    """
    all_embeddings = []
    all_labels = []
    user_names = []

    # Get list of user directories
    user_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    print(f"Found {len(user_dirs)} users.")

    for user_id, user_name in enumerate(user_dirs):
        user_path = os.path.join(root_path, user_name)
        npy_files = glob.glob(os.path.join(user_path, "*.npy"))
        
        user_data_pool = []
        
        # Load all data for this user
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                # Ensure data is 2D (N_samples, Embedding_dim)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                user_data_pool.append(data)
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

        if not user_data_pool:
            print(f"No valid data found for user: {user_name}")
            continue

        # Concatenate all arrays for this user
        full_user_data = np.concatenate(user_data_pool, axis=0)
        
        # Random sampling
        num_available = full_user_data.shape[0]
        if num_available >= samples_per_user:
            indices = np.random.choice(num_available, samples_per_user, replace=False)
            sampled_data = full_user_data[indices]
        else:
            print(f"Warning: User {user_name} has only {num_available} samples (requested {samples_per_user}). Using all.")
            sampled_data = full_user_data

        all_embeddings.append(sampled_data)
        all_labels.extend([user_name] * sampled_data.shape[0])
        user_names.append(user_name)

    if not all_embeddings:
        raise ValueError("No data loaded.")

    X = np.concatenate(all_embeddings, axis=0)
    y = np.array(all_labels)
    
    return X, y, user_names

def plot_tsne(X, y, user_names):
    save_dir=r"D:\CodeSpace\PROJECT\UltraPrint\data\show_results"
    """
    Performs t-SNE reduction to 2D and 3D and plots the results.
    """
    print(f"Starting t-SNE on data shape: {X.shape}...")

    # 2D t-SNE
    print("Computing 2D t-SNE...")
    tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)

    # 3D t-SNE
    print("Computing 3D t-SNE...")
    tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
    X_3d = tsne_3d.fit_transform(X)

    # Setup colors
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))

    # Plot 2D
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        mask = (y == label)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.7, s=30, c=[color_map[label]])
    plt.title('t-SNE Visualization (2D)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_2d.png'))  
    plt.show()
    plt.close()
    

    # Plot 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_labels:
        mask = (y == label)
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], label=label, alpha=0.7, s=30, c=[color_map[label]])
    ax.set_title('t-SNE Visualization (3D)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_3d.png'))
    plt.show()
    plt.close()





if __name__ == "__main__":
    # Replace this with your actual data path
    DATA_ROOT = r"D:\CodeSpace\PROJECT\UltraPrint\data\test_data" 
    
    # Create dummy data for demonstration if path doesn't exist (Optional check)
    if not os.path.exists(DATA_ROOT):
        print(f"Path {DATA_ROOT} not found. Please ensure the path is correct.")
    else:
        try:
            X, y, users = load_and_sample_data(DATA_ROOT, samples_per_user=100)
            plot_tsne(X, y, users)
        except Exception as e:
            print(f"An error occurred: {e}")