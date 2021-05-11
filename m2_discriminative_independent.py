import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class M2(nn.Module):

    def __init__(self, dim_x=28*28, dim_z=100, dim_y=10, gamma=1):
        super(M2, self).__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.gamma = gamma

        self.theta_d = nn.Sequential(nn.Linear(dim_x, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, dim_y))
        self.theta_g = nn.Sequential(nn.Linear(dim_z+dim_y, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, 2*dim_x))
                                    
        self.phi_z = nn.Sequential(nn.Linear(dim_x+dim_y, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, 2*dim_z))

        self.phi_y = nn.Sequential(nn.Linear(dim_x, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, dim_y))

    def forward(self, x_l_p, y_l, x_u_p):
        """
        x_l, x_u: [0, 1] normalized pixel entries
        y_l: one-hot label
        """
        
        ## LABELLED PART ##

        # Binarize labelled images
        x_l = torch.bernoulli(x_l_p)

        # Compute posterior of z:  q(z | x, y)
        qz_l_param = self.phi_z(torch.cat([x_l, y_l], dim=1))
        qz_l_mu = qz_l_param[:, :self.dim_z]
        qz_l_log_sigma_sq = qz_l_param[:, self.dim_z:]

        # Sample from z posterior
        z_l_sample = self._draw_sample(qz_l_mu, qz_l_log_sigma_sq)

        # Compute p(x | y, z)
        px_l_param = self.theta_g(torch.cat([z_l_sample, y_l], dim=1))
        px_l_mu = px_l_param[:, :self.dim_x]
        px_l_log_sigma_sq = px_l_param[:, self.dim_x:]

        # Compute L_l(x, y) in Eq (2)
        L_l = self._L(x_l, px_l_mu, px_l_log_sigma_sq, y_l, z_l_sample, qz_l_mu, qz_l_log_sigma_sq)
        L_l = L_l.mean(1).sum()

        # Discriminator loss
        ypred_l_logits = self.phi_y(x_l)

        ypred_l_disriminative_logits = self.theta_d(x_l)

        # D_l = - F.cross_entropy(ypred_l_logits, y_l.argmax(1), reduction='none')
        D_l = - F.cross_entropy(ypred_l_discriminative_logits, y_l.argmax(1), reduction='none')
        
        D_l = D_l.sum()   
        
        L_prior = 0.

        
        
        ## UNLABELLED PART ##


        # Binarize unlabelled images
        x_u = torch.bernoulli(x_u_p)

        # Estimate logits of q(y | x)
        y_u_logits = self.phi_y(x_u)
        y_u = F.softmax(y_u_logits, dim=1)

        # Compute L_l(x, yhat) in Eq (3)
        L_lhat = torch.zeros(y_u.shape)

        for label in range(self.dim_y):

            yhat = torch.diag(self.dim_y)[label]

            # Compute posterior of z:  q(z | x, yhat)
            qz_u_param = self.phi_z(torch.cat([x_u, yhat], dim=1))
            qz_u_mu = qz_u_param[:, :self.dim_z]
            qz_u_log_sigma_sq = qz_u_param[:, self.dim_z:]
            
            # Sample from z posterior
            z_u_sample = self._draw_sample(qz_u_mu, qz_u_log_sigma_sq)

            # Compute p(x | yhat, z)
            px_u_param = self.theta_g(torch.cat([z_u_sample, yhat], dim=1))
            px_u_mu = px_u_param[:, :self.dim_x]
            px_u_log_sigma_sq = px_u_param[:, self.dim_x:]

            # Compute L_l(x, yhat) in Eq (2)
            _L_lhat = self._L(x_u, px_u_mu, px_u_log_sigma_sq, yhat, z_u_sample, qz_u_mu, qz_u_log_sigma_sq)
            _L_lhat = _L_lhat.mean(1)
            L_lhat[:, label] = _L_lhat      

        # Compute L_U(x) in Eq (3)
        L_u = y_u * (L_lhat - torch.log(y_u))
        L_u = L_u.mean(1).sum()


        ## TOTAL L ##


        # Compute L(x, y) in Eq (4)
        L_tot = L_l + self.gamma * D_l + L_u
        loss = - L_tot


        return loss 

    def _draw_sample(self, mu, log_sigma_sq):
        
        epsilon = torch.randn(mu.size())
        sample = mu + ( torch.exp( 0.5 * log_sigma_sq ) * epsilon )

        return sample

    def _L(self, x_l, px_mu, px_log_sigma_sq, y_l, z_sample, qz_mu, qz_log_sigma_sq):
        
        y_prior = (1. / self.dim_y) * torch.ones(y_l.size())

        log_prior_y = - F.cross_entropy(y_prior, y_l.argmaz(1), reduction='none')

        log_lik =  utils.normal_logpdf(x_l, px_mu, px_log_sigma_sq)

        log_prior_z = utils.stdnormal_logpdf(z_sample)

        log_post_z = utils.normal_logpdf(z_sample, qz_mu, qz_log_sigma_sq)

        return log_prior_y + log_lik + log_prior_z - log_post_z