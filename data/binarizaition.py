import os
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

class ITQ:
    def __init__(self, num_bits=256, num_iter=50):
        self.num_bits = num_bits
        self.num_iter = num_iter
        self.pca = PCA(n_components=num_bits)
        self.R = None  # Rotation matrix

    def fit(self, X):
        """
        Fit the ITQ model:
        1. PCA reduction
        2. Iterative quantization to find optimal rotation R
        """
        # 1. PCA Reduction， PCA降维，不能低于其 帧和维度 中的最小值
        print("Fitting PCA...")
        X_pca = self.pca.fit_transform(X)
        print(f"PCA reduced shape: {X_pca.shape}")

        # 2. ITQ Iteration
        print("Fitting ITQ Rotation...")
        # Initialize random rotation matrix
        R = np.random.randn(self.num_bits, self.num_bits)
        U, _, Vt = np.linalg.svd(R)
        R = U.dot(Vt)
        
        for i in range(self.num_iter):
            # Project data
            V = X_pca.dot(R)
            # Quantize (B is the binary matrix, -1 or 1)
            B = np.sign(V)
            
            # Update Rotation
            # Minimize ||B - VR||^2 -> Maximize tr(B * R^T * V^T)
            # This is the Orthogonal Procrustes problem
            M = B.T.dot(X_pca)
            U, _, Vt = np.linalg.svd(M)
            R = Vt.T.dot(U.T)
            
        self.R = R
        return self

    def transform(self, X):
        """
        Apply PCA and Rotation, then binarize.
        Returns packed bits (uint8) or raw binary (-1/1 or 0/1) depending on need.
        Here we return 0/1 float or int representation for simplicity in .npy storage.
        """
        X_pca = self.pca.transform(X)
        V = X_pca.dot(self.R)
        # Binarize: maps >=0 to 1, <0 to 0
        B = np.where(V >= 0, 1, 0).astype(np.int8)
        return B

def get_all_npy_files(root_dir):
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy') and not file.endswith('_bin.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def main():
    # Configuration
    data_root = r'D:/CodeSpace/PROJECT/UltraPrint/data/test_data' # Replace with your actual data path
    target_dim = 256
    
    # 1. Collect files
    print(f"Scanning files in {data_root}...")
    all_files = get_all_npy_files(data_root)
    if not all_files:
        print("No .npy files found.")
        return

    # 2. Load a subset of data to train ITQ (PCA + Rotation)
    # Loading all data might be too large for memory, so we sample.
    print("Loading training data for ITQ...")
    train_data = []
    sample_count = 0
    max_samples = 50000 # Adjust based on your memory
    
    # Simple sampling strategy: take first few frames from files until limit
    for fpath in tqdm(all_files):
        try:
            data = np.load(fpath)
            # Ensure data is 2D (frames, dim)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Take a few frames for training
            n_frames = data.shape[0]
            take_n = min(n_frames, 10) 
            train_data.append(data[:take_n])
            sample_count += take_n
            
            if sample_count >= max_samples:
                break
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not train_data:
        print("Failed to load training data.")
        return

    X_train = np.vstack(train_data)
    print(f"Training data shape: {X_train.shape}")

    # 3. Train ITQ
    itq = ITQ(num_bits=target_dim, num_iter=50)
    itq.fit(X_train)

    # 4. Process all files and save
    print("Processing all files...")
    for fpath in tqdm(all_files):
        try:
            # Load original embedding
            X = np.load(fpath)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Transform to binary
            X_bin = itq.transform(X)
            
            # Construct output filename
            # /path/to/video.npy -> /path/to/video_bin.npy
            out_path = fpath.replace('.npy', '_bin.npy')
            
            # Save
            np.save(out_path, X_bin)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()