import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        output_dims,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.output_dims = output_dims
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(device)
        device = next(self.parameters()).device
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters())
        
        for epoch in range(max_epochs):
            self.train()
            total_loss = 0
            reconstruction_loss = 0
            kl_loss = 0
            
            for batch in train_loader:
                X = batch[0]
                optimizer.zero_grad()
                
                z_mean, z_log_var, z = self.encoder(X)
                reconstruction = self.decoder(z)
                
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                
                # Normalize the loss by the batch size
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                kl_loss += kl.item()
            
            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs} | Total loss: {total_loss / len(train_loader):.4f} | "
                    f"Recon loss: {reconstruction_loss / len(train_loader):.4f} | "
                    f"KL loss: {kl_loss / len(train_loader):.4f}")
    
    def encode(self, X, **kwargs):
        X = X.permute(0, 2 ,1)
        z_mean, z_log_var, z = self.encoder(X, zero_var=False)
        return z_mean
    
    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        return x_decoded
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(next(self.parameters()).device)
            z_mean, z_log_var, z = self.encoder(X)
            x_decoded = self.decoder(z_mean)
        return x_decoded.cpu().detach().numpy()
    
    def decode(self, z_mean, random_score=0, *kwargs):
        return self.decoder(z_mean)
    
    def shift_augmentation(self, X, random_shift_ratio):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            X = torch.FloatTensor(X).to(device)
            random_shift = torch.randn(len(X), self.output_dims).to(device)
            z_mean, z_log_var, z = self.encoder(X)
            z_mean += random_shift * random_shift_ratio
            x_decoded = self.decoder(z_mean)
        return x_decoded
    
    def sphere_augmentation(self, X, theta=0.05, norm=True):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            X = torch.FloatTensor(X).to(device)
            # random_shift = torch.randn(len(X), self.output_dims).to(device)
            z_mean, z_log_var, z = self.encoder(X)
            reprs = self.sample_nearby_on_sphere(z_mean, theta, z_log_var, norm)
            x_decoded = self.decoder(reprs)
            x_decoded = x_decoded / (torch.std(x_decoded, dim=-1, keepdim=True) + 1e-7)
        return x_decoded
    
    @staticmethod
    def sample_nearby_on_sphere(mu, theta=0.1, z_log_var=None, norm=False):
        # mu: (batch_size, d)
        # z_log_var: (batch_size, d) 或 None
        # theta: 标量或张量

        # 确保 theta 是与 mu 类型和设备一致的张量
        theta = torch.tensor(theta, dtype=mu.dtype, device=mu.device)
        batch_size, d = mu.shape

        if norm:
            # 为每个样本生成随机 theta 值
            theta = torch.randn(batch_size, dtype=mu.dtype, device=mu.device) * theta
        else:
            # 将标量 theta 扩展为与 batch_size 相同的形状
            theta = theta.expand(batch_size)

        # 将 theta 转换为弧度
        theta *= torch.pi * 2

        # 生成与 mu 相同类型和设备的随机扰动向量 r
        r = torch.randn(batch_size, d, dtype=mu.dtype, device=mu.device)
        if z_log_var is not None:
            r = torch.exp(0.5 * z_log_var) * r

        # 计算正交扰动向量 v
        mu_dot_mu = torch.sum(mu * mu, dim=1, keepdim=True)  # (batch_size, 1)
        r_dot_mu = torch.sum(r * mu, dim=1, keepdim=True)    # (batch_size, 1)
        projection = (r_dot_mu / mu_dot_mu) * mu             # (batch_size, d)
        v = r - projection                                   # (batch_size, d)
        v_norm = torch.norm(v, dim=1, keepdim=True)          # (batch_size, 1)
        v_unit = v / v_norm                                  # (batch_size, d)

        # 计算 cos(theta) 和 sin(theta)
        cos_theta = torch.cos(theta).unsqueeze(1)            # (batch_size, 1)
        sin_theta = torch.sin(theta).unsqueeze(1)            # (batch_size, 1)

        # # 可选：打印每个样本的 cos(theta) 和 sin(theta)
        # for i in range(batch_size):
        #     print(cos_theta[i].item(), sin_theta[i].item())

        # 计算新的潜在向量 z_new
        z_new = mu * cos_theta + v_unit * sin_theta          # (batch_size, d)

        return z_new
    
    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.output_dims).to(device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss

    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth"))

    def load_weights(self, model_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth")))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "output_dims": self.output_dims,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

if __name__ == "__main__":
    pass