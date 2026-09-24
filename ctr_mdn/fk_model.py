import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTRForwardMDN(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, n_components=5, output_dim=7, diag_floor=1e-3):
        super().__init__()
        self.K = n_components
        self.D = output_dim
        self.diag_floor = diag_floor

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components * output_dim)
        self.fc_U = nn.Linear(hidden_dim, n_components * output_dim * output_dim)
        self.act = nn.ReLU()

        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for layer in (self.fc_pi, self.fc_mu, self.fc_U):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        B = x.shape[0]
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))

        logits = self.fc_pi(h)
        mu = self.fc_mu(h).view(B, self.K, self.D)
        U = self.fc_U(h).view(B, self.K, self.D, self.D)
        diag = F.softplus(torch.diagonal(U, dim1=-2, dim2=-1)) + self.diag_floor
        U = torch.triu(U, diagonal=1) + torch.diag_embed(diag)
        return logits, mu, U

    @staticmethod
    def covariance(U, eps=1e-6):
        P = U.transpose(-1, -2) @ U
        P = P + eps * torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        return torch.linalg.inv(P)


def mdn_nll(logits, mu, U, y, eps=1e-8):
    """Negative log-likelihood of y under the predicted Gaussian mixture."""
    D = mu.shape[-1]
    z = U @ (y.unsqueeze(1) - mu).unsqueeze(-1)
    maha = (z ** 2).sum(dim=(-2, -1))
    log_det = torch.log(torch.diagonal(U, dim1=-2, dim2=-1) + eps).sum(-1)
    log_comp = log_det - 0.5 * D * math.log(2 * math.pi) - 0.5 * maha
    return -torch.logsumexp(F.log_softmax(logits, dim=-1) + log_comp, dim=1).mean()
