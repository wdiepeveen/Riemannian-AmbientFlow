import torch
import torch.nn as nn

from src.diffeomorphisms import Diffeomorphism

class UnbendDiffeomorphism(Diffeomorphism):
    def __init__(self, angle, delta) -> None:
        super().__init__(2)
        self.rot = torch.tensor([
            [torch.cos(torch.tensor([angle])), -torch.sin(torch.tensor([angle]))], 
            [torch.sin(torch.tensor([angle])), torch.cos(torch.tensor([angle]))]
            ])
        self.delta = delta

    def huber_loss(self, p):
        """
        Computes the huber loss.
        :param p: Tensor of shape (N,), where N is the batch size.
        :return: Transformed tensor of shape (N,).
        """
        return torch.abs(p) * (p.abs() > self.delta) + (1/(2 * self.delta) * p **2 + 1/2 * self.delta) * (p.abs() <= self.delta)
    
    def derivative_huber_loss(self, p):
        """
        Computes the derivative of the huber loss.
        :param p: Tensor of shape (N,), where N is the batch size.
        :return: Transformed tensor of shape (N,).
        """
        return torch.sign(p) * (p.abs() > self.delta) + (1/self.delta * p) * (p.abs() <= self.delta)

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y = x.clone()
        y = torch.einsum("ab,Nb->Na", self.rot, y)
        y[:, 0] = y[:, 0] - self.huber_loss(y[:, 1])
        y[:, 1] *= 2 ** (1/2)
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 2), where N is the batch size.
        :return: Inverted tensor of shape (N, 2).
        """
        x = y.clone()
        x[:, 0] = y[:, 0] + self.huber_loss(2 ** (-1/2) * y[:, 1])
        x[:, 1] *= 2 ** (-1/2)
        x = torch.einsum("ab,Na->Nb", self.rot, x)
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        D_x = X.clone()
        D_x = torch.einsum("ab,Nb->Na", self.rot, D_x)
        D_x[:, 0] = D_x[:, 0] - self.derivative_huber_loss(torch.einsum("ab,Nb->Na", self.rot, x)[:, 1]) * D_x[:, 1]
        D_x[:,1] *= 2 ** (1/2)
        return D_x

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 2), inputs.
        :param Y: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        D_y = Y.clone()
        D_y[:, 0] = D_y[:, 0] + 2 ** (-1/2) * self.derivative_huber_loss(2 ** (-1/2) * y[:, 1]) * D_y[:, 1]
        D_y[:, 1] *=  2 ** (-1/2)
        D_y = torch.einsum("ab,Na->Nb", self.rot, D_y)
        return D_y

