import torch
from torch.autograd.functional import vjp
import math
from nflows import flows, distributions

from src.inverse_problems import InverseProblem

class EncoderDecoderProblem(InverseProblem):
    def __init__(self, data_shape, forward_operator, context_encoder, flow_transform, noise_level, num_samples=10, low_rank_reg_param=None, clean_data_reg_param=None):
        super().__init__(math.prod(data_shape), forward_operator)
        self.data_shape = data_shape
        self.sigma = noise_level
        self.M = num_samples
        self.mu_1 = low_rank_reg_param
        self.mu_2 = clean_data_reg_param

        self.context_encoder = context_encoder
        self.flow_transform = flow_transform

        self.p_pri = flows.Flow(transform=self.flow_transform, distribution=distributions.StandardNormal(shape=self.data_shape))
        self.p_pos = flows.Flow(transform=self.flow_transform, distribution=distributions.ConditionalDiagonalNormal(shape=self.data_shape, context_encoder=self.context_encoder))


    def reconstruction_loss(self, y, x=None):
        # Initialize loss components
        logavgexp_terms = []
        for _ in range(self.M):
            # Sample from p_posterior(x|y)
            x_reconstructed = self.p_pos.sample(1, context=y)[:,0]

            # Compute individual terms in the logavgexp expression
            log_p_pri = self.p_pri.log_prob(x_reconstructed) 
            log_q_n = -(1/(2 * self.sigma**2) * (self.A.forward(x_reconstructed) - y)**2).sum(dim=tuple(range(1,y.dim())))
            log_p_pos = self.p_pos.log_prob(x_reconstructed, context=y)

            logavgexp_terms.append(log_p_pri + log_q_n - log_p_pos)


        # Compute logavgexp over all samples
        loss = -(torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()

        # Regularization
        if self.mu_1 is not None:
            D_0_phi_inv = vjp(lambda x: self.p_pri._transform(x, context=None)[0],
                              (torch.zeros(self.d, self.d, device=y.device).reshape(self.d, *self.data_shape),), 
                              (torch.eye(self.d, self.d, device=y.device).reshape(self.d, *self.data_shape),)
                              )[1][0].reshape(self.d, self.d)
            loss += self.mu_1 * torch.linalg.norm(D_0_phi_inv, ord='fro')

        if self.mu_2 is not None and x is not None:
            log_p_pri_gt = self.p_pri.log_prob(x)
            loss -= self.mu_2 * log_p_pri_gt.mean()

        return loss
    
    def reconstruct(self, y): # TODO now we get both mean and variance. We need to get only the mean from the encoder
        return self.flow_transform(self.context_encoder(y))