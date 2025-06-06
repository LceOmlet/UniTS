# ts_url/losses/tnc_loss.py
import torch
import torch.nn as nn

class TncLoss(nn.Module):
    def __init__(self, w=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = w  # 加权项

    def forward(self, z_t, z_p, z_n):
        """
        z_t: anchor 表示 (batch_size, feat_dim)
        z_p: 邻域表示 (batch_size, feat_dim)
        z_n: 非邻域表示 (batch_size, feat_dim)
        """
        # 判别器输出用 cosine similarity 近似
        pos_score = torch.cosine_similarity(z_t, z_p)
        neg_score = torch.cosine_similarity(z_t, z_n)

        # 标签
        ones = torch.ones_like(pos_score)
        zeros = torch.zeros_like(neg_score)

        # 真实 vs 错误 vs 混淆损失
        p_loss = self.bce(pos_score, ones)
        n_loss = self.bce(neg_score, zeros)
        n_loss_u = self.bce(neg_score, ones)

        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2
        return loss
