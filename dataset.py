import torch
# import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class USDataset(Dataset):
    def __init__(self, csv_file, mode = None):
        #输入的语音梅尔频谱长度为80*501 和对于标签路径
        print("dataset path: ",csv_file)
        self.data = pd.read_csv(csv_file)

        self.data1 = self.data.iloc[:]['ultrasound_path'].tolist()
        self.data2 = self.data.iloc[:]['video_feature_path'].tolist()
        # self.data3 = self.data.iloc[:]['audio_feature_path'].tolist()
        self.data4 = self.data.iloc[:]['TrueUser'].tolist()
        self.data5 = self.data.iloc[:]['label'].tolist()  


    def split_samples(self, data):
        new_data_paths = []
        # Process data paths
        for idx in range(len(data)):
            path = data.iloc[idx, 0]
            prefix = '/'.join(path.split('/')[:-1]) + '/'
            basename = path.split('/')[-1].split('.')[0]  #xxx_acc
            suffix = '.' + path.split('.')[-1]
            
            for i in range(0, 7):
                new_path = prefix + basename[:-4] + f'_part{i}' + basename[-4:] + suffix   #xxx_parti_acc.npy
                new_data_paths.append(new_path)
        
        return new_data_paths

    def check_forbidden_samples(self, data, forbidden_path):
        forbidden_samples = set()
        try:
            with open(forbidden_path, 'r') as f:
                forbidden_samples = {line.strip() for line in f}
        except FileNotFoundError:
            print("Warning: forbidden_train_samples.txt not found")

        # Filter out forbidden samples by comparing base names without path and extension
        filtered_data = data.copy()
        if forbidden_samples:
            # Extract base names from full paths (remove path and extension)
            filtered_data['basename'] = filtered_data.iloc[:, 0].apply(lambda x: x.split('\\')[-1].split('.')[0])
            # Keep only rows where basename is not in forbidden_samples
            filtered_data = filtered_data[~filtered_data['basename'].isin(forbidden_samples)]
            # Drop the temporary basename column
            filtered_data = filtered_data.drop('basename', axis=1)
            
        return filtered_data
    
    def get_label_lists(self, label):
        label_dict = {}
        for i in range(0, len(label)):
            path = label.iloc[i, 0]
            name = path.split('/')[-1].split('.')[0]
            label_dict[name] = path
        return label_dict
    
    def __len__(self):
        return len(self.data2)

    def get_audio_path(self, idx):
        return self.data2[idx]
        # return self.data.iloc[idx, 0] 
    
    def get_audio_name(self, idx):
        return self.data2[idx].split('/')[-1].split('.')[0]
        # return self.data.iloc[idx, 0].split('/')[-1].split('.')[0]
    
    def __getitem__(self, idx):
    
        us_path = self.data1[idx]
        video_path = self.data2[idx]
        # audio_path = self.data_audio[idx]
        true_user = self.data4[idx]
        label = self.data5[idx]   
        #=======================================================#
        #us data 
        us_data = np.load(us_path)  #shape : [2, 32, 3000*T]
        us_data = us_data[0, :, :] #取第一个接收通道  shape : [32, 3000*T]
        us_data = torch.tensor(us_data, dtype=torch.float32)


        #=======================================================#

        #video data 
        video_data = np.load(video_path) #shape : [10*T, feature_dim] = [10*T, 512]
        video_data = torch.tensor(video_data, dtype=torch.float32)


        #=======================================================#
        #audio data
        # audio_data = np.load(audio_path) #shape : [10*T, feature_dim] = [10*T, 512]   
        # audio_data = torch.tensor(audio_data, dtype=torch.float32)
        #=======================================================#   
        true_user = int(true_user)
        true_user = torch.tensor(true_user, dtype=torch.long)

        #=======================================================#   
        label = int(label)
        label = torch.tensor(label, dtype=torch.long)
        #=======================================================#
        # show samples
        if idx < 5 :
            if not os.path.exists('./results'):
                os.makedirs('./results')
            # Plot both spectrograms in a single figure
            plt.figure(figsize=(15, 5))
            
            for i in range(4):
                plt.subplot(4, 1, i+1)
                plt.plot(us_data[i, :].numpy())
                plt.title(f'Channel {i}')
            
            plt.tight_layout()
            plt.savefig(f'./results/sample_show_{idx}.png')
            plt.close()
                
        #shape:
        # input: ultrasound [32, 3000*T]  
        # output: video [10*T, feature_dim]

        return {
            'ultrasound':us_data.contiguous(), #ultrasound waveform data
            'video': video_data.contiguous(), #video feature
            # 'audio': audio_data.contiguous(), #audio feature  
            'TrueUser': true_user,  #true user label    
            'Label': label,  #user label
        }