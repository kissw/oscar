# Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Dataloader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os, sys
import const
from config import Config
config = Config.neural_net

from torchvision import transforms
from PIL import Image

class CropImage(object):
    def __init__(self, crop_x1, crop_y1, crop_x2, crop_y2):
        self.crop_x1 = crop_x1
        self.crop_y1 = crop_y1
        self.crop_x2 = crop_x2
        self.crop_y2 = crop_y2

    def __call__(self, img):
        return img.crop((self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2))
    
class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=True, flip=False):
        """
        Args:
            csv_file (string): CSV 파일의 경로.
            root_dir (string): 모든 이미지가 있는 디렉토리.
            transform (callable, optional): 샘플에 적용될 선택적 변환.
        """
        self.frame = pd.read_csv(csv_file, header=None, usecols=[0, 12, 13, 14, 15])
        self.root_dir = root_dir + '/'
        self.flip = flip
        crop_transform = CropImage(Config.data_collection['image_crop_x1'],Config.data_collection['image_crop_y1'],Config.data_collection['image_crop_x2'],Config.data_collection['image_crop_y2']
        )
        if transform is True:
            self.transform = transforms.Compose([crop_transform,
                                                 transforms.Resize((config['input_image_height'], config['input_image_width'])),
                                                 transforms.ToTensor()])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])

        tar_img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        tar_str = torch.tensor(self.frame.iloc[idx, 2], dtype=torch.float32)
        tar_vel = torch.tensor(self.frame.iloc[idx, 3], dtype=torch.float32)
        tar_time = torch.tensor(self.frame.iloc[idx, 4], dtype=torch.float32)
        
        with Image.open(img_name) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)

        with Image.open(tar_img_name) as tar_img:
            tar_img = tar_img.convert('RGB')
            if self.transform:
                tar_img = self.transform(tar_img)

        if self.flip:
            flipped_img = TF.hflip(img) # 100% 확률로 반전
            flipped_tar_img = TF.hflip(tar_img)
            flipped_tar_str = -tar_str  # steering angle 반대 방향으로 조정

            if self.transform:
                flipped_img = self.transform(flipped_img)
                flipped_tar_img = self.transform(flipped_tar_img)

            combined_images = torch.cat([img.unsqueeze(0), flipped_img.unsqueeze(0)], dim=0)
            combined_tar_images = torch.cat([tar_img.unsqueeze(0), flipped_tar_img.unsqueeze(0)], dim=0)
            combined_str = torch.tensor([tar_str, flipped_tar_str])

            sample = {'image': combined_images, 'tar_image': combined_tar_images, 'tar_str': combined_str, 'tar_vel': tar_vel, 'tar_time': tar_time}
        else:
            sample = {'image': img, 'tar_image': tar_img, 'tar_str': tar_str, 'tar_vel': tar_vel, 'tar_time': tar_time}# 차원 출력

        return sample

import matplotlib.pyplot as plt
import torchvision
import numpy as np

class InternalModel(nn.Module):
    def __init__(self, batch_size=100):
        super(InternalModel, self).__init__()
        self.batch_size = batch_size
        self.img_rows = 160
        self.img_cols = 160
        self.img_channels = 3
        self.latent_dim = 50
        self.hidden_dim = 500
        self.fwd_r_factor = self.img_rows * self.img_cols * 100
        self.fwd_kl_factor = 10
        self.fwd_d_factor = 10
        self.figures = {}

        # Encoder architecture
        self.vae_encoder = nn.Sequential(
            nn.Conv2d(self.img_channels, 24, kernel_size=5, padding='same'),
            nn.BatchNorm2d(24), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same'),
            nn.BatchNorm2d(36), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(36, 48, kernel_size=5, padding='same'),
            nn.BatchNorm2d(48), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64), nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(102400,500)
        self.layer_norm = nn.LayerNorm(500)
        self.tanh = nn.Tanh()
        self.bilstm = nn.LSTM(input_size=500, hidden_size=self.hidden_dim, batch_first=True, bidirectional=True)
        self.linear2 = nn.Linear(1000, 500)  # LSTM이 양방향이므로 hidden_size*2
        self.elu1 = nn.ELU()
        self.linear3 = nn.Linear(500, 100)
        self.elu2 = nn.ELU()

        # Latent
        self.fc_mean = nn.Linear(100, self.latent_dim)
        self.fc_logvar = nn.Linear(100, self.latent_dim)

        # Dense layers for additional inputs
        self.fc_str = nn.Sequential(nn.Linear(1, 100),
                                    nn.BatchNorm1d(100),
                                    nn.ELU(),
                                    nn.Linear(100, 50),
                                    nn.BatchNorm1d(50),
                                    nn.ELU())
        self.fc_vel = nn.Sequential(nn.Linear(1, 100),
                                    nn.BatchNorm1d(100),
                                    nn.ELU(),
                                    nn.Linear(100, 50),
                                    nn.BatchNorm1d(50),
                                    nn.ELU())
        self.fc_time = nn.Sequential(nn.Linear(1, 100),
                                    nn.BatchNorm1d(100),
                                    nn.ELU(),
                                    nn.Linear(100, 50),
                                    nn.BatchNorm1d(50),
                                    nn.ELU())
        # Decoder architecture
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim+50*3, 64 * 40 * 40), #C * H * W
            nn.ELU(),
            nn.Unflatten(1, (64, 40, 40)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(48, 36, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(36, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Discriminator architecture
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.img_channels, 24, kernel_size=5, padding='same'),
            nn.BatchNorm2d(24), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same'),
            nn.BatchNorm2d(36), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(36, 48, kernel_size=5, padding='same'),
            nn.BatchNorm2d(48), nn.LeakyReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

        # Optimizers
        self.optimizer_fwd = optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.0000001)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=0.00005, weight_decay=0.0000001)

    def plot_images(self, images, epoch, batch, title, window_id, num_images=10):
        selected_images = images[:num_images]
        img_grid = torchvision.utils.make_grid(selected_images, nrow=num_images)
        np_img = img_grid.numpy()

        if window_id not in self.figures:
            self.figures[window_id] = plt.figure(figsize=(15, 4))

        plt.figure(self.figures[window_id].number)
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.title(f'{title}')
        plt.axis('off')
        plt.tight_layout()

        # 100 batch마다 이미지 저장
        if window_id == 0:
            if batch % 1000 == 0:
                save_path = f'/home2/kdh/vae/new_dataset/vae_latent_compute/result/e{epoch}_b{batch}.png'
                plt.savefig(save_path)
        # 1 epoch마다 이미지 저장
        if window_id == 1:
            save_path = f'/home2/kdh/vae/new_dataset/vae_latent_compute/result/e{epoch}.png'
            plt.savefig(save_path)

        plt.close(self.figures[window_id])


    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward_vae(self, img, str, vel, time):
        batch_size, timesteps, C, H, W = img.size()
        img = img.view(batch_size * timesteps, C, H, W)
        img = self.vae_encoder(img)
        img = self.flatten(img)
        img = self.linear1(img)
        img = self.layer_norm(img)
        img = self.tanh(img)
        img = img.view(batch_size, timesteps, -1)
        lstm_output, _ = self.bilstm(img)
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.linear2(lstm_output)
        lstm_output = self.elu1(lstm_output)
        lstm_output = self.linear3(lstm_output)
        lstm_output = self.elu2(lstm_output)
        z_mean = self.fc_mean(lstm_output)
        z_logvar = self.fc_logvar(lstm_output)
        sampling = self.reparameterize(z_mean, z_logvar)
        str = str.view(batch_size, 1)
        vel = vel.view(batch_size, 1)
        time = time.view(batch_size, 1)
        str = self.fc_str(str)
        vel = self.fc_vel(vel)
        time = self.fc_time(time)
        concat = torch.cat([sampling, str, vel, time], dim=1)
        decoded = self.vae_decoder(concat)
        return decoded, z_mean, z_logvar

    def vae_r_loss(self, y_true, y_pred):
        r_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        return self.fwd_r_factor * r_loss
    def vae_kl_loss(self, z_mean, z_logvar):
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return self.fwd_kl_factor * kl_loss
    def vae_d_loss(self, fake_output):# 가짜(vae) 이미지에 대한 판별자 손실
        fake_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
        return self.fwd_d_factor * fake_loss

    def forward_discriminator(self, x):
        return self.discriminator(x)
    def discriminator_loss(self, real_output, fake_output):# 가짜(vae) 이미지에 대한 판별자 손실
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def train_data_loader(self, datapath, batch_size=32, num_workers=16, transform=True):
        if datapath[-1] == '/':
            datapath = datapath[:-1]

        loc_slash = datapath.rfind('/')
        if loc_slash != -1:  # there is '/' in the data path
            model_name = datapath[loc_slash + 1:]  # get folder name
        else:
            model_name = datapath

        csv_path = datapath + '/' + model_name + const.DATA_EXT 
        dataset = DrivingDataset(csv_file=os.path.join(csv_path), root_dir=datapath, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Train function (placeholder, needs proper implementation)
    def train_model(self, train_data_loader, epochs=100000, device=None):
        min_loss = float('inf')  # 이전 에포크의 최소 손실을 저장할 변수 초기화

        for epoch in range(epochs):
            epoch_vae_loss = 0.0
            epoch_disc_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
                img, tar_img, tar_str, tar_vel, tar_time = [x.to(device) for x in data.values()]
                img = img.unsqueeze(1)
                # Training loop implementation here
                self.optimizer_fwd.zero_grad()
                self.optimizer_disc.zero_grad()
                # Forward pass through VAE
                reconstructed_img, z_mean, z_logvar = self.forward_vae(img, tar_str, tar_vel, tar_time)
                # Compute VAE losses
                r_loss = self.vae_r_loss(tar_img, reconstructed_img)
                kl_loss = self.vae_kl_loss(z_mean, z_logvar)
                # Forward pass through discriminator for fake images
                fake_output = self.forward_discriminator(reconstructed_img)
                # Compute discriminator loss for VAE
                vae_d_loss = self.vae_d_loss(fake_output)
                # VAE loss is sum of reconstruction, KL divergence, and discriminator losses
                vae_loss = r_loss + kl_loss + vae_d_loss
                vae_loss.backward()
                self.optimizer_fwd.step()
                # print(r_loss, kl_loss, vae_d_loss)
                # Compute discriminator losses for real and fake images
                real_output = self.forward_discriminator(tar_img)
                fake_output = self.forward_discriminator(reconstructed_img.detach())
                disc_loss = self.discriminator_loss(real_output, fake_output)
                disc_loss.backward()
                self.optimizer_disc.step()

                epoch_vae_loss += vae_loss.item()
                epoch_disc_loss += disc_loss.item()
                # 일정 반복마다 현재 손실 출력
                if (i+1) % 100000 == 0:
                    # print(f'[{epoch + 1}, {i+1}] vae: {epoch_vae_loss / (i+1):.3f}, disc: {epoch_disc_loss / (i+1):.3f}')
                    cur_output = f'\r[{epoch + 1}, {i + 1}] vae: {epoch_vae_loss / (i + 1):.3f}, disc: {epoch_disc_loss / (i + 1):.3f}'
                    sys.stdout.write(cur_output)
                    sys.stdout.flush()
                    with torch.no_grad():
                        reconstructed_img, _, _ = self.forward_vae(img, tar_str, tar_vel, tar_time)
                    self.plot_images(reconstructed_img[:10].cpu(), epoch+1, i+1, f'Batch {i+1}', window_id=0)

            # 현재 에포크의 평균 손실 계산
            avg_epoch_loss = (epoch_vae_loss + epoch_disc_loss) / len(train_data_loader)
            if avg_epoch_loss < min_loss:
                min_loss = avg_epoch_loss
                torch.save(self.state_dict(), f'/home2/kdh/vae/new_dataset/vae_latent_compute/fwd_e{epoch}_l{min_loss}.pth')
                with torch.no_grad():
                    reconstructed_img, _, _ = self.forward_vae(img, tar_str, tar_vel, tar_time)
                self.plot_images(reconstructed_img[:10].cpu(), epoch+1, 1, f'Epoch {epoch+1}', window_id=1)
            sys.stdout.write('\n')
            # print(f'Epoch {epoch+1}, vae: {epoch_vae_loss / len(train_data_loader):.4f}, disc: {epoch_disc_loss / len(train_data_loader):.4f}')
            cur_output = f'\rEpoch {epoch+1}, vae: {epoch_vae_loss / len(train_data_loader):.4f}, disc: {epoch_disc_loss / len(train_data_loader):.4f}'
            sys.stdout.write(cur_output)
            sys.stdout.write('\n')

def load_model_for_training(model, checkpoint_path, train_data_loader, epochs, device):
    # 모델 상태 로드
    model.load_state_dict(torch.load(checkpoint_path))

    # 모델을 훈련 모드로 설정
    model.train()

    # 이전에 저장된 최소 손실 값 찾기
    min_loss = float('inf')

    # 추가 훈련 루프
    for epoch in range(epochs):
        epoch_vae_loss = 0.0
        epoch_disc_loss = 0.0

        for i, data in enumerate(train_data_loader, 0):
            img, tar_img, tar_str, tar_vel, tar_time = [x.to(device) for x in data.values()]
            img = img.unsqueeze(1)

            model.optimizer_fwd.zero_grad()
            model.optimizer_disc.zero_grad()

            reconstructed_img, z_mean, z_logvar = model.forward_vae(img, tar_str, tar_vel, tar_time)
            r_loss = model.vae_r_loss(tar_img, reconstructed_img)
            kl_loss = model.vae_kl_loss(z_mean, z_logvar)
            fake_output = model.forward_discriminator(reconstructed_img)
            vae_d_loss = model.vae_d_loss(fake_output)

            vae_loss = r_loss + kl_loss + vae_d_loss
            vae_loss.backward()
            model.optimizer_fwd.step()

            real_output = model.forward_discriminator(tar_img)
            fake_output = model.forward_discriminator(reconstructed_img.detach())
            disc_loss = model.discriminator_loss(real_output, fake_output)
            disc_loss.backward()
            model.optimizer_disc.step()

            epoch_vae_loss += vae_loss.item()
            epoch_disc_loss += disc_loss.item()

            if (i+1) % 1000 == 0:
                cur_output = f'\r[{epoch + 1}, {i + 1}/{len(train_data_loader)}] vae: {epoch_vae_loss / (i + 1):.3f}, disc: {epoch_disc_loss / (i + 1):.3f}'
                sys.stdout.write(cur_output)
                sys.stdout.flush()
                with torch.no_grad():
                    reconstructed_img, _, _ = model.forward_vae(img, tar_str, tar_vel, tar_time)
                model.plot_images(reconstructed_img[:10].cpu(), epoch+1, i+1, f'Batch {i+1}', window_id=0)

        avg_epoch_loss = (epoch_vae_loss + epoch_disc_loss) / len(train_data_loader)
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(model.state_dict(), f'/home2/kdh/vae/new_dataset/vae_latent_compute/fwd_e{epoch+1}_l{min_loss}.pth')
            with torch.no_grad():
                reconstructed_img, _, _ = model.forward_vae(img, tar_str, tar_vel, tar_time)
            model.plot_images(reconstructed_img[:10].cpu(), epoch+1, 1, f'Epoch {epoch+1}', window_id=1)
        sys.stdout.write('\n')
        cur_output = f'\rEpoch {epoch+1}, vae: {epoch_vae_loss / len(train_data_loader):.4f}, disc: {epoch_disc_loss / len(train_data_loader):.4f}'
        sys.stdout.write(cur_output)
        sys.stdout.write('\n')

# GPU 설정

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 40
i = InternalModel(batch_size=batch_size)
i = i.to(device)
train_loader = i.train_data_loader('/home2/kdh/vae/new_dataset/2023-08-22-17-26-03/', batch_size=batch_size, num_workers=24)

load_model = True

# 사용 예
if load_model:
    checkpoint_path = '/home2/kdh/vae/new_dataset/vae_latent_compute/fwd_e0_l4677.181148159427.pth'
    load_model_for_training(i, checkpoint_path, train_loader, 100000, device)
else:
    i.train_model(train_loader, device=device)

