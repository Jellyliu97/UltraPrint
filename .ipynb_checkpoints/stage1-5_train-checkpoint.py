#训练超声信号的编码器，用于和视频、声音进行对齐，使用对比学习损失函数

import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import random
import numpy as np
import time
# from networks.vision_transformer import SwinUnet as ViT_seg

# from utils import DiceLoss
# from config import get_config
# from WG_dataset2 import PosNegDataset
from dataset import USDataset
import csv
# from model.encoder import DualImageContrastiveModel
# from model.accVoice_swinUnet import SwinTransformerSys as SwinUnet
# from model.CrossViT3 import ImageCrossViT
from model.stage1_model1 import WaveformModel
# import networks.accVoice_vitUnet as VitUnet
from utils.loss import InfoNCELoss2, InfoNCELoss, BatchInfoNCELoss
from matplotlib import pyplot as plt

#single GPU
ngpu = 1
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#multi GPU
gpus = [0,1]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

def load_stage1_frozen_params(model, checkpoint_path):
    """
    Load pretrained parameters for the all modules (speech_encoder, speech_mlp, acc_encoder and fusion_network)
    from a Stage 1 .pth checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping weight loading.")
        return

    print(f"Loading Stage 1 weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint saving formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("Checkpoint keys:", list(state_dict.keys())[:10])  # Print first 10 keys for debugging

    # Load Waveform Encoder
    # Filter keys that belong to waveform_encoder and remove the prefix
    waveform_encoder_dict = {k.replace('encoder.', ''): v 
                        for k, v in state_dict.items() 
                        if k.startswith('encoder.')}
    
    if waveform_encoder_dict:
        msg = model.encoder.load_state_dict(waveform_encoder_dict, strict=True)
        print(f"Successfully loaded Waveform Encoder: {msg}")
    else:
        print("Warning: No Waveform Encoder weights found in checkpoint.")


    # # 2. Load Speech MLP
    # speech_mlp_dict = {k.replace('module.mlp_speech.', ''): v
    #                     for k, v in state_dict.items()
    #                     if k.startswith('module.mlp_speech.')}
    # if speech_mlp_dict:
    #     msg = model.mlp_speech.load_state_dict(speech_mlp_dict, strict=True)
    #     print(f"Successfully loaded Speech MLP: {msg}")
    # else:
    #     print("Warning: No Speech MLP weights found in checkpoint.")


def trainer_synapse(model):
   
    learning_rate = 1e-3 #学习率
    num_epochs = 400
    batch_size = 512
    
    #正样本和负样本，负样本是正样本的10倍数量，在每一个epoch中随机选择1/10的负样本，然后和正样本结合组成一个epoch的训练数据
    # data_path = r"E:\dataset\ultrasound_video_audio\DATA\dataset\train_dataset.csv" #正样本 windows
    data_path = "/root/autodl-tmp/UltraPrint_dataset/dataset_npy/train_dataset.csv" #正样本 linux
    positive_dataset = USDataset(data_path)

    print("====== Train positive Dataset Count: =======", positive_dataset.__len__())

    # 1. 将两个数据集合并
    # full_dataset = ConcatDataset([positive_dataset, negative_dataset])

    # # 2. 创建动态负样本采样器
    # class NegativeSampler:
    #     def __init__(self, negative_dataset_length, sample_ratio=0.1):
    #         self.negative_length = negative_dataset_length
    #         self.sample_ratio = sample_ratio

    #     def __iter__(self):
    #         # 每个epoch都会调用，生成新的随机索引
    #         sample_size = int(self.negative_length * self.sample_ratio)
    #         # 生成负样本部分的随机索引（在合并数据集中，负样本位于正样本之后）
    #         negative_indices = torch.randperm(self.negative_length)[:sample_size] + len(positive_dataset)
    #         # 正样本的索引（全部使用）
    #         positive_indices = torch.arange(len(positive_dataset))
    #         # 合并索引并打乱顺序
    #         all_indices = torch.cat([positive_indices, negative_indices])
    #         shuffled_indices = all_indices[torch.randperm(len(all_indices))]
    #         return iter(shuffled_indices.tolist())

    #     def __len__(self):
    #         # 返回一个epoch的总步数（样本数）
    #         return len(positive_dataset) + int(self.negative_length * self.sample_ratio)

    # 初始化采样器
    # sampler = NegativeSampler(len(negative_dataset), sample_ratio=0.1)

    # 3. 创建DataLoader，注意shuffle=False因为采样器会处理顺序
    # train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler) #每个epoch动态采样负样本

    train_loader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True) #静态采样，直接打乱整个数据集进行训练

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #增加collate_fn函数处理同一个batch中label不一样长的问题
    # dataloader = [next(iter(dataloader)) for _ in range(10)] #取前 x 个batch
    
    
    model = model.to(device) #single gpu
    # model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0]) #multi gpu

    #finetune
    pretrain_model_path = r"./checkpoints/stage1/stage1_best_model.pth"
    load_stage1_frozen_params(model, pretrain_model_path)
    
    
    # criterion = nn.MSELoss()
    # criterion = InfoNCELoss() #对比损失函数
    #上下两路分别提取各自得特征，中间进行融合两者得特征对判别正负样本得下路进行指导。上路不需要指导，上路是为了提取用户语音的特征信息
    # critertion_feature = nn.CosineSimilarity(dim=1)  #计算两个特征向量的余弦相似度
    # critertion_crossFusion = BatchInfoNCELoss()  #两者中间融合得特征进行对比损失计算，正样本和负样本。最终这部分特征会指导判别正负样本
    # critertion_contrastive = InfoNCELoss()
    # critertion_BCE = nn.BCEWithLogitsLoss()  #二分类交叉熵损失函数
    critertion_MSE1 = nn.MSELoss()
    critertion_MSE2 = nn.MSELoss()
    


    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate) ###
    optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), # 使用filter进行过滤
    lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=32,eta_min=1e-6)

    # writer = SummaryWriter(snapshot_path + '/log')
    # iter_num = 0
    max_epoch = num_epochs
    max_iterations = max_epoch * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=100)
    best_loss = 10e10
    
    model.train()
    all_losses = []
    for epoch in iterator:
        # batch_loss = 0
        for i_batch, batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch}", total=len(train_loader),
                                           leave=False):
            ultrasound = batch['ultrasound']    #[N, 32，3000*T] batch_size, 
            video_feature = batch['video']        #[N, 50，512]
            # speaker = batch['speaker']
            # user = batch['user']

            # ultrasound = torch.transpose(ultrasound, -1, -2)  #转置，变成[N, 3000*T, 1] 适配模型尺寸
            # video_feature = torch.transpose(video_feature, -1, -2)      #转置，变成[N, 512, 50]
            # audio = audio.to(device=device, dtype=torch.float32)  #GPU上训练，数据必须放入GPU，参数在GPU上更新
            # label = label.to(device=device, dtype=torch.float32)
            ultrasound = ultrasound.cuda(non_blocking=True) #将 Tensor 数据从 CPU 移动到 GPU 时，可以通过设置 non_blocking=True 来启用异步传输模式，从而提高计算效率。
            video_feature = video_feature.cuda(non_blocking=True)  #GPU上训练，数据必须放入GPU，参数在GPU上更新, non_blocking=True启用异步数据传输。

            feature_u, recon_v, recon_u = model(ultrasound, video_feature)

            # outputs= model(audio) 
            # print("output shape: ",outputs.shape)
            loss1 = critertion_MSE1(recon_u, feature_u)  #计算特征向量和说话人向量的余弦相似度
            loss2 = critertion_MSE2(recon_v, video_feature)  #计算音频特征和加速度特征的对比损失
        

            # loss2 = critertion_crossFusion(cross_feature, user)  #计算融合特征和正负样本标签的对比损失,同一个批次内部，利用正负样本的标签进行对比学习
            # loss3 = critertion_BCE(user_speaker.squeeze(), user.float())  #计算二分类交叉熵损失函数
            # loss = (1-loss1).mean() + loss2 + loss3  #总损失函数， loss2 *0.1系数，平衡损失
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
        #     iter_num = iter_num + 1
        #     # writer.add_scalar('info/lr', lr_, iter_num)
        #     writer.add_scalar('info/total_loss', loss, iter_num)
        #     # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
        #     #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
        #     batch_loss += loss.item()
        # batch_loss /= len(train_loader)
        # logging.info('Train epoch: %d : loss : %f' % (
        #     epoch, batch_loss))
        print(f'mse Loss1: {loss1.item():.4f}, contrastive Loss2: {loss2.item():.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        all_losses.append([epoch+1, loss.item()])     

        if (epoch + 1) % 10 == 0:
            scheduler.step() #动态更新学习率,每10个epoch更新一次学习率
            torch.save(model.state_dict(), f'./checkpoints/stage1-5/stage1-5_model_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')

        if epoch == 0 or loss.item() < best_loss: #训练集上损失最小，不代表在测试集上效果最好。一般取的是验证集准确度最高或者验证集损失最小进行存储
            best_loss = loss.item()
            torch.save(model.state_dict(), './checkpoints/stage1-5/stage1-5_best_model.pth') #多卡训练时，应该改成 torch.save(model.module.state_dict(), './outputs/best_model.pth')
            print(f'Best model saved with loss {best_loss}')

        # Save the training loss to a CSV file
        # with open('./output/train_process.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([epoch + 1, loss.item()])
    with open('./outputs/stage1-5/stage1-5_train_process.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        writer.writerows(all_losses)

    # writer.close()
    return "Training Finished!"


def main_train_WaveformModel():
    # 创建模型
    T = 5 # 5 seconds
    batch_size = 4
    feature_dim = 512
    frames_per_sec = 10
    
    model = WaveformModel(feature_dim=feature_dim)

    trainer_synapse(model)


def test_model(model):
    
    batch_size = 256

    #test dataset path
    #正样本和负样本，负样本是正样本的10倍数量，合在一起进行推理
    # data_path =  r"E:\dataset\ultrasound_video_audio\DATA\dataset\test_dataset.csv" #测试样本
    data_path =  r"E:\dataset\ultrasound_video_audio\DATA\dataset\test_dataset_ldxTest.csv" #测试样本


    test_dataset = USDataset(data_path)
    print("====== Test Dataset Count: =======", test_dataset.__len__())

    #output path
    output_path = r"E:\dataset\ultrasound_video_audio\record\stage1_test_dataset" #mismatching negitive samples, 只保存预测的label

    # 加载预训练模型参数
    # model_path = "./checkpoints/model_epoch_400.pth"  # 使用最后一个epoch的模型
    model_path = "./checkpoints/stage1/stage1_best_model.pth"  # 使用best_model
    # model_path = "Z:/dataset/accelerometer/record/RESULT_original_xyzTrans_removeTimbre_average_BIGVGAN_swinUnet_LGDZN/model_epoch_400.pth"  # 使用best_model
    # model_path = r"Z:\dataset\accelerometer_audio\AccVoice\record\RESULT_original_removeTimbre_average_BIGVGAN_swinUnet_crossUser2\best_model.pth"

    model = model.to(device) #single gpu test
    # model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0]) #multi gpu test
    model.load_state_dict(torch.load(model_path))


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sample_paths = [test_dataset.get_audio_path(i) for i in range(len(test_dataset))] #读取原始音频路径
    sample_names = [test_dataset.get_audio_name(i) for i in range(len(test_dataset))]
    
    model.eval()
    with torch.no_grad():
        save_path_index = 0
        for batch in test_loader:
            ultrasound = batch['ultrasound']    #[N, 1, 3000*T] batch_size, 
            video = batch['video'] #[N, 50, 512]

            # ultrasound = torch.transpose(ultrasound, -2, -1)  #转置，变成[N, 1, 500, 80]
            # video = torch.transpose(video, -2, -1)
            # audio = audio.to(device=device, dtype=torch.float32)  #GPU上训练，数据必须放入GPU，参数在GPU上更新
            # label = label.to(device=device, dtype=torch.float32)
            ultrasound = ultrasound.cuda(non_blocking=True)  #GPU上训练，数据必须放入GPU，参数在GPU上更新, non_blocking=True启用异步数据传输。
            video = video.cuda(non_blocking=True) #将 Tensor 数据从 CPU 移动到 GPU 时，可以通过设置 non_blocking=True 来启用异步传输模式，从而提高计算效率。
            
            time_start = time.time()
            feature_u, proj_feature_u, ultrasound_reconstructed = model(ultrasound)
            # fake_clean = model(noisy_imgs)
            time_end = time.time()

            print("inference time: ", time_end - time_start)
            print("average inference time per sample: ", (time_end - time_start)/ultrasound_reconstructed.shape[0])

            output = feature_u.squeeze(dim=1).cpu().numpy()  #N, 50, 512
            output1 = proj_feature_u.cpu().numpy()  #N, 512 
            output2 = ultrasound_reconstructed.squeeze(dim=1).cpu().numpy() #N, 32, 3000*T
            
            print("output shape: ", output1.shape)
            for j in range(output1.shape[0]):
                if save_path_index >= len(sample_names):  # Check if we've reached the end of the dataset
                    break
                print(f"Processing sample {save_path_index + 1}/{len(sample_names)}: {sample_names[save_path_index]}")
           
                parts = sample_paths[save_path_index].split('\\')
                # folder1 = parts[-3]
                folder2 = parts[-2]
     
                # Create nested directory structure
                # nested_dir = os.path.join(output_path, folder1, folder2)
                nested_dir = os.path.join(output_path, folder2)
                os.makedirs(nested_dir, exist_ok=True)

                # Full output path with nested directories
                output_file = os.path.join(nested_dir, f"{sample_names[save_path_index]}_mid.npy")
                output_file1 = os.path.join(nested_dir, f"{sample_names[save_path_index]}_midPool.npy")
                output_file2 = os.path.join(nested_dir, f"{sample_names[save_path_index]}_us.npy")



                out = output[j] #存储ultrasound 中间特征
                out1 = output1[j] #存储ultrasound 中间特征池化后的特征
                out2 = output2[j] #存储重建的ultrasound 波形    
                np.save(output_file, out)
                np.save(output_file1, out1)
                np.save(output_file2, out2)

                save_path_index += 1

def main_test_WaveformModel():
   # 创建模型
    T = 3 # 5 seconds
    batch_size = 4
    feature_dim = 512
    frames_per_sec = 10
    
    model = WaveformModel(feature_dim=feature_dim)

    test_model(model)

if __name__ == "__main__":
    main_train_WaveformModel()
    # main_test_WaveformModel()


