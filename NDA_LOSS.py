import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, MelScale
import numpy as np

class SISNRLoss(nn.Module):
    def __init__(self):
        super(SISNRLoss, self).__init__()

    def forward(self, enhanced_signal, clean_signal):
        min_length = min(enhanced_signal.size(-1), clean_signal.size(-1))
        enhanced_signal = enhanced_signal[:, :min_length]
        clean_signal = clean_signal[:, :min_length]

        enhanced_mean = torch.mean(enhanced_signal, dim=-1, keepdim=True)
        clean_mean = torch.mean(clean_signal, dim=-1, keepdim=True)
        enhanced_signal = enhanced_signal - enhanced_mean
        clean_signal = clean_signal - clean_mean

        dot_product = torch.sum(enhanced_signal * clean_signal, dim=-1, keepdim=True)
        target_energy = torch.sum(clean_signal ** 2, dim=-1, keepdim=True)
        s_target = (dot_product / (target_energy + 1e-8)) * clean_signal

        e_noise = enhanced_signal - s_target

        s_target_energy = torch.sum(s_target ** 2, dim=-1)
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1)
        si_snr = 10 * torch.log10(s_target_energy / (e_noise_energy + 1e-8))

        return -torch.mean(si_snr)

from utils.v_activlev import v_activlev

class NDALoss(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=256, n_mels=80, gamma=5, mu=20, omega=20, eta=3, tau=2):
        super(NDALoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gamma = gamma
        self.mu = mu
        self.omega = omega
        self.eta = eta
        self.tau = tau

        # Spectrogram and Mel-scale transforms
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1)
        self.mel_scale = MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

    def to(self, device):
        super().to(device)
        self.spectrogram = self.spectrogram.to(device)
        self.mel_scale = self.mel_scale.to(device)
        return self

    def compute_mel_spectrum(self, waveform):
        spectrogram = self.spectrogram(waveform)  # Compute spectrogram
        mel_spectrum = self.mel_scale(spectrogram)  # Convert to MEL-scale
        return torch.pow(mel_spectrum, 1/3)  # Apply cube root compression
    def compute_differential_spectrum(self, mel_spectrum):
        alpha = torch.tensor([0.25, 0.5, 0.75, 1.0], device=mel_spectrum.device)
        velocity_diff = torch.zeros_like(mel_spectrum)
        for k in range(1, 5):
            velocity_diff[:, :, :-k] += alpha[k-1] * (mel_spectrum[:, :, k:] - mel_spectrum[:, :, :-k])

        beta = torch.tensor([-0.357193, -0.607143, -0.285714, 0.25, 1.0], device=mel_spectrum.device)
        acceleration_diff = torch.zeros_like(mel_spectrum)
        for k in range(5):
            if k == 0:
                acceleration_diff += beta[k] * mel_spectrum
            else:
                acceleration_diff[:, :, :-k] += beta[k] * (mel_spectrum[:, :, k:] - mel_spectrum[:, :, :-k])

        return velocity_diff, acceleration_diff

    def envelope_loss(self, B_c, B_s, V):
        """
        Envelope Loss Function
        :param B_c: Enhanced speech MEL spectrum (cube root compressed) [batch, n_mels, time_frames]
        :param B_s: Clean speech MEL spectrum (cube root compressed) [batch, n_mels, time_frames]
        :param V: Voice Activity Detection (VAD) information [batch, 1, time_samples]
        :return: Envelope loss
        """
        # Downsample V to match the time frames of B_c and B_s
        time_frames = B_c.size(2)  # Number of time frames in B_c and B_s
        V_downsampled = F.avg_pool1d(V, kernel_size=V.size(2) // time_frames, stride=V.size(2) // time_frames)
        V_downsampled = V_downsampled.squeeze(1)  # Remove the channel dimension

        # Expand V_downsampled to match the shape of B_c and B_s
        V_downsampled = V_downsampled.unsqueeze(1).expand_as(B_c)  # [batch, n_mels, time_frames]

        # Compute the asymmetric difference
        diff = B_c - B_s
        asym_diff = torch.where(diff > 0, diff, self.eta * diff)

        # Compute the loss
        loss = torch.mean(torch.abs(asym_diff * V_downsampled))

        return loss

    def continuity_loss_speech(self, B_vel_c, B_vel_s, B_acc_c, B_acc_s, V):
        # Downsample V to match the time frames of B_c and B_s
        time_frames = B_vel_c.size(2)  # Number of time frames in B_c and B_s
        V_downsampled = F.avg_pool1d(V, kernel_size=V.size(2) // time_frames, stride=V.size(2) // time_frames)
        V_downsampled = V_downsampled.squeeze(1)  # Remove the channel dimension

        # Expand V_downsampled to match the shape of B_c and B_s
        V = V_downsampled.unsqueeze(1).expand_as(B_vel_c)  # [batch, n_mels, time_frames]
        loss_vel = torch.mean(torch.abs(B_vel_c - B_vel_s) * V)
        loss_acc = torch.mean(torch.abs(B_acc_c - B_acc_s) * V)
        return loss_vel + loss_acc

    def continuity_loss_non_speech(self, B_c, V_s):
        # Downsample V to match the time frames of B_c and B_s
        time_frames = B_c.size(2)  # Number of time frames in B_c and B_s
        V_downsampled = F.avg_pool1d(V_s, kernel_size=V_s.size(2) // time_frames, stride=V_s.size(2) // time_frames)
        V_downsampled = V_downsampled.squeeze(1)  # Remove the channel dimension

        # Expand V_downsampled to match the shape of B_c and B_s
        V_s = V_downsampled.unsqueeze(1).expand_as(B_c)  # [batch, n_mels, time_frames]

        loss = 0
        for tau in range(-self.tau, self.tau + 1):
            if tau == 0:
                continue
            shifted_B_c = torch.roll(B_c, shifts=tau, dims=2)
            loss += torch.mean(torch.abs(B_c - shifted_B_c) * V_s)
        return loss

    def forward(self, enhancedwav, cleanwav):
        B_c = self.compute_mel_spectrum(enhancedwav)
        B_s = self.compute_mel_spectrum(cleanwav)

        B_vel_c, B_acc_c = self.compute_differential_spectrum(B_c)
        B_vel_s, B_acc_s = self.compute_differential_spectrum(B_s)

        # 处理每个批次的VAD
        batch_size = cleanwav.shape[0]
        V = []
        for i in range(batch_size):
            # 获取单个音频并转换为numpy数组
            clean_np = cleanwav[i].numpy()
            clean_np = np.array(clean_np,dtype=np.float32)
            _, _, _, vad = v_activlev(sp=clean_np.copy(), fs=16000, mode='0')  # 使用copy()创建数组副本
            V.append(vad)
        # 将VAD列表转换为tensor并调整维度
        V = np.array(V, dtype=np.float32)
        V = torch.tensor(V, dtype=torch.float32, device=cleanwav.device) # [batch_size, time]
        V = V.unsqueeze(1)  # [batch_size, 1, time]

        V_s = 1 - V  # Shape: [batch_size, 1, time_frames]

        L_env = self.envelope_loss(B_c, B_s, V)
        L_sp = self.continuity_loss_speech(B_vel_c, B_vel_s, B_acc_c, B_acc_s, V)
        L_sil = self.continuity_loss_non_speech(B_c, V_s)

        SISNR_Loss = SISNRLoss()
        denoising_loss = SISNR_Loss(enhancedwav, cleanwav)

        L_NDA = denoising_loss + self.gamma * L_env + self.mu * L_sp + self.omega * L_sil
        return L_NDA


if __name__ == "__main__":
    loss_fn = NDALoss(sample_rate=16000, n_fft=512, hop_length=256, n_mels=80, gamma=5, mu=20, omega=20, eta=3, tau=2)

    enhancedwav = torch.randn(2, 32000)
    cleanwav = torch.randn(2, 32000)

    loss = loss_fn(enhancedwav, cleanwav)
    print("Loss: ", loss.item())
