import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np


# class ThetaD(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ThetaD, self).__init__()

#         ## phi_y ##
#         self.phi_y1 = nn.Linear(in_dim, 500)
#         self.phi_y2 = nn.Linear(500, 500)
#         self.phi_y3 = nn.Linear(500, out_dim)

#         self.phi_y = nn.Sequential( self.phi_y1, nn.ReLU(inplace=True),
#                                     self.phi_y2, nn.ReLU(inplace=True),
#                                     self.phi_y3)
        
#         ## epsilon ##
#         self.epsilon1 = nn.Linear(in_dim, 500)
#         self.epsilon2 = nn.Linear(500, 500)
#         self.epsilon3 = nn.Linear(500, out_dim)

#         self.epsilon_list = [self.epsilon1, nn.ReLU(inplace=True), self.epsilon2, nn.ReLU(inplace=True), self.epsilon3]

#     def forward(self, x):

#         with torch.no_grad():
#             x_p1 = self.phi_y1(x)
#         x_e1 = self.epsilon1(x)
#         x1 = F.relu(x_p1 + x_e1)

#         with torch.no_grad():
#             x_p2 = self.phi_y2(x1)
#         x_e2 = self.epsilon2(x1)
#         x2 = F.relu(x_p2 + x_e2)

#         with torch.no_grad():
#             x_p3 = self.phi_y3(x2)
#         x_e3 = self.epsilon3(x2)
#         x3 = x_p3 + x_e3

#         return x3     

class ThetaD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThetaD, self).__init__()

        ## phi_y ##
        self.phi_y1 = nn.Linear(in_dim, 500)
        self.phi_y2 = nn.Linear(500, 500)
        self.phi_y3 = nn.Linear(500, out_dim)

        self.phi_y = nn.Sequential( self.phi_y1, nn.ReLU(inplace=True),
                                    self.phi_y2, nn.ReLU(inplace=True),
                                    self.phi_y3)
        
        ## epsilon ##
        self.epsilon1 = nn.Linear(in_dim, 500)
        self.epsilon2 = nn.Linear(500, 500)
        self.epsilon3 = nn.Linear(500, out_dim)

        self.epsilon_list = [self.epsilon1, nn.ReLU(inplace=True), self.epsilon2, nn.ReLU(inplace=True), self.epsilon3]

    def forward(self, x):

        x_p1 = self.phi_y1(x)
        x_e1 = self.epsilon1(x)
        x1 = F.relu(x_p1 + x_e1)

        x_p2 = self.phi_y2(x1)
        x_e2 = self.epsilon2(x1)
        x2 = F.relu(x_p2 + x_e2)

        x_p3 = self.phi_y3(x2)
        x_e3 = self.epsilon3(x2)
        x3 = x_p3 + x_e3

        return x3       

class Blended(nn.Module):

    def __init__(   self,
                    device, gamma,
                    num_batches, batch_size,
                    dim_x=28*28, dim_z=100, dim_y=10):
        super(Blended, self).__init__()

        self.device = device

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.gamma = gamma

        self.num_batches = num_batches
        self.batch_size = batch_size

        # self.theta_g = nn.Sequential(nn.Linear(dim_z+dim_y, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                              nn.Linear(500, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                              nn.Linear(500, dim_x))
                                    
        # self.phi_z = nn.Sequential(nn.Linear(dim_x+dim_y, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                            nn.Linear(500, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                            nn.Linear(500, 2*dim_z))

        # self.phi_y = nn.Sequential(nn.Linear(dim_x, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                            nn.Linear(500, 500), nn.BatchNorm1d(500, affine=True), nn.ReLU(inplace=True),
        #                            nn.Linear(500, dim_y))
        
        # self.theta_d = nn.Sequential(nn.Linear(dim_x, 500), nn.ReLU(inplace=True),
        #                              nn.Linear(500, 500), nn.ReLU(inplace=True),
        #                              nn.Linear(500, dim_y))

        self.theta_d = ThetaD(dim_x, dim_y)


        self.theta_g = nn.Sequential(nn.Linear(dim_z+dim_y, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, dim_x))
                                    
        self.phi_z = nn.Sequential(nn.Linear(dim_x+dim_y, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, 500), nn.ReLU(inplace=True),
                                   nn.Linear(500, 2*dim_z))

        # self.phi_y = nn.Sequential(nn.Linear(dim_x, 500), nn.ReLU(inplace=True),
        #                            nn.Linear(500, 500), nn.ReLU(inplace=True),
        #                            nn.Linear(500, dim_y))
        self.phi_y = self.theta_d.phi_y

        assert id(self.phi_y == self.theta_d.phi_y)

    def forward(self, x_l_p, y_l, x_u_p):
        """
        x_l, x_u: [0, 1] normalized pixel entries
        y_l: one-hot label
        """
        
        ## LABELLED PART ##
        # self.eval()

        # Binarize labelled images
        with torch.no_grad():
            x_l = torch.bernoulli(x_l_p)
        # x_l = x_l_p

        # Compute posterior of z:  q(z | x, y)
        # qz_l_param = self.phi_z(torch.cat([x_l, torch.argmax(y_l, dim=1, keepdim=True).float()], dim=1))
        qz_l_param = self.phi_z(torch.cat([x_l, y_l], dim=1))
        qz_l_mu = qz_l_param[:, :self.dim_z]
        qz_l_log_sigma_sq = qz_l_param[:, self.dim_z:]

        # Sample from z posterior
        z_l_sample = self._draw_sample(qz_l_mu, qz_l_log_sigma_sq)

        # Compute p(x | y, z)
        # px_l_param = self.theta_g(torch.cat([z_l_sample, torch.argmax(y_l, dim=1, keepdim=True).float()], dim=1))
        # px_l_param = self.theta_g(torch.cat([z_l_sample, y_l], dim=1))
        # px_l_mu = px_l_param[:, :self.dim_x]
        # px_l_log_sigma_sq = px_l_param[:, self.dim_x:]
        px_l_param = self.theta_g(torch.cat([z_l_sample, y_l], dim=1))
        px_l_mu = torch.sigmoid(px_l_param)

        # Compute L_l(x, y) in Eq (2)
        L_l = self._L(x_l, px_l_mu, y_l, z_l_sample, qz_l_mu, qz_l_log_sigma_sq)
        # print("Labelled")
        L_l = L_l.sum()

        # Discriminator term
        # ypred_l_logits = self.phi_y(x_l)
        ypred_l_disriminative_logits = self.theta_d(x_l)

        D_l = - F.cross_entropy(ypred_l_disriminative_logits, y_l.argmax(1), reduction='none')
        
        D_l = D_l.sum()  

        ## PRIOR PART ##
        
        L_prior = 0.

        # for i,weight in enumerate(self.theta_d):
        #     if 'Linear' in str(weight.type):
        #         alpha = 0.001
        #         lambda_d = (1-alpha)/(alpha)
        #         flattened_weight = torch.cat([weight.weight.reshape((-1,)), weight.bias.reshape((-1,))], dim=0)
        #         flattened_mean = torch.cat([self.phi_y[i].weight.reshape((-1,)), self.phi_y[i].bias.reshape((-1,))], dim=0)
        #         flattened_sigma_sq = (torch.ones(flattened_mean.size()) / lambda_d ).to(self.device)
        #         flattened_log_sigma_sq = torch.log(flattened_sigma_sq)
        #         L_prior += utils.normal_logpdf(flattened_weight, flattened_mean, flattened_log_sigma_sq).sum()
                # d = flattened_weight.shape[0]
                # L_prior += (d/2)*np.log(lambda_d) + utils.stdnormal_logpdf(np.sqrt(lambda_d)*(flattened_weight - flattened_mean)).sum()

        for i, weight in enumerate(self.theta_d.epsilon_list):
            if 'Linear' in str(weight.type):
                alpha = 0.001
                lambda_d = (1-alpha)/(alpha)
                flattened_weight = torch.cat([weight.weight.reshape((-1,)), weight.bias.reshape((-1,))], dim=0)
                # flattened_mean = torch.cat([self.phi_y[i].weight.reshape((-1,)), self.phi_y[i].bias.reshape((-1,))], dim=0)
                flattened_mean = torch.zeros(flattened_weight.size()).to(self.device)
                flattened_sigma_sq = (torch.ones(flattened_mean.size()) / lambda_d ).to(self.device)
                flattened_log_sigma_sq = torch.log(flattened_sigma_sq)
                L_prior += utils.normal_logpdf(flattened_weight, flattened_mean, flattened_log_sigma_sq).sum()
                # d = flattened_weight.shape[0]
                # L_prior += (d/2)*np.log(lambda_d) + utils.stdnormal_logpdf(np.sqrt(lambda_d)*(flattened_weight - flattened_mean)).sum()            
        
        for i,weight in enumerate(self.theta_g):
            if 'Linear' in str(weight.type):
                flattened_weight = torch.cat([weight.weight.reshape((-1,)), weight.bias.reshape((-1,))], dim=0)
                L_prior += utils.stdnormal_logpdf(flattened_weight).sum()

        for i,weight in enumerate(self.phi_z):
            if 'Linear' in str(weight.type):
                flattened_weight = torch.cat([weight.weight.reshape((-1,)), weight.bias.reshape((-1,))], dim=0)
                L_prior += utils.stdnormal_logpdf(flattened_weight).sum()
        for i,weight in enumerate(self.phi_y):
            if 'Linear' in str(weight.type):
                flattened_weight = torch.cat([weight.weight.reshape((-1,)), weight.bias.reshape((-1,))], dim=0)
                L_prior += utils.stdnormal_logpdf(flattened_weight).sum()
 
        # print(L_prior)
        
        
        ## UNLABELLED PART ##
        # self.train()

        # Binarize unlabelled images
        with torch.no_grad():
            x_u = torch.bernoulli(x_u_p)
        # x_u = x_u_p

        # Estimate logits of q(y | x)
        y_u_logits = self.phi_y(x_u)
        y_u = F.softmax(y_u_logits, dim=1)

        # Compute L_l(x, yhat) in Eq (3)
        L_lhat = torch.zeros(y_u.shape).to(self.device)

        for label in range(self.dim_y):

            yhat = torch.eye(self.dim_y)[label].unsqueeze(0).to(self.device)
            yhat = yhat.expand(x_u.size(0), -1)

            # Compute posterior of z:  q(z | x, yhat)
            # qz_u_param = self.phi_z(torch.cat([x_u, torch.argmax(yhat, dim=1, keepdim=True).float()], dim=1))
            qz_u_param = self.phi_z(torch.cat([x_u, yhat], dim=1))
            qz_u_mu = qz_u_param[:, :self.dim_z]
            qz_u_log_sigma_sq = qz_u_param[:, self.dim_z:]
            
            # Sample from z posterior
            z_u_sample = self._draw_sample(qz_u_mu, qz_u_log_sigma_sq)

            # Compute p(x | yhat, z)
            # px_u_param = self.theta_g(torch.cat([z_u_sample, torch.argmax(yhat, dim=1, keepdim=True).float()], dim=1))
            # px_u_param = self.theta_g(torch.cat([z_u_sample, yhat], dim=1))
            # px_u_mu = px_u_param[:, :self.dim_x]
            # px_u_log_sigma_sq = px_u_param[:, self.dim_x:]
            px_u_param = self.theta_g(torch.cat([z_u_sample, yhat], dim=1))
            px_u_mu = torch.sigmoid(px_u_param)

            # Compute L_l(x, yhat) in Eq (2)
            _L_lhat = self._L(x_u, px_u_mu, yhat, z_u_sample, qz_u_mu, qz_u_log_sigma_sq)
            _L_lhat = _L_lhat.unsqueeze(1)

            if label == 0:
                L_lhat = _L_lhat
            else:
                L_lhat = torch.cat([L_lhat, _L_lhat], dim=1)

        # Compute L_U(x) in Eq (3)
        # print(L_lhat)
        assert L_lhat.size() == y_u.size()
        L_u = y_u * (L_lhat - torch.log(y_u))
        L_u = L_u.sum(1).sum()


        ## TOTAL L ##


        # Compute L(x, y) in Eq (4)
        L_tot = L_l +  D_l + L_u 
        # print(L_tot)
        L_tot = (L_tot*self.num_batches + L_prior)
        # print(L_prior)
        # print('###',L_tot)
        loss = - L_tot / (self.batch_size*self.num_batches)


        return loss 

    def _draw_sample(self, mu, log_sigma_sq):
        
        epsilon = torch.randn(mu.size()).to(self.device)
        sample = mu + ( torch.exp( 0.5 * log_sigma_sq ) * epsilon )

        return sample

    def _L(self, x_l, px_mu, y_l, z_sample, qz_mu, qz_log_sigma_sq):
        
        y_prior = (1. / self.dim_y) * torch.ones(y_l.size()).to(self.device)
        log_prior_y = - F.cross_entropy(y_prior, y_l.argmax(1), reduction='none')

        log_lik =  utils.bernoulli_logpdf(x_l, px_mu)
        # log_prior_z = utils.stdnormal_logpdf(z_sample)
        log_prior_z = utils.gaussian_marg(qz_mu, qz_log_sigma_sq)
        
        # log_post_z = utils.normal_logpdf(z_sample, qz_mu, qz_log_sigma_sq)
        log_post_z = utils.gaussian_ent(qz_log_sigma_sq)

        # print(torch.max(px_mu), torch.min(px_mu))
        # print(log_prior_y.sum()/self.batch_size , log_lik.sum(1).sum()/self.batch_size , log_prior_z.sum(1).sum()/self.batch_size, - log_post_z.sum(1).sum()/self.batch_size)

        return log_prior_y + log_lik.sum(1) + log_prior_z.sum(1) - log_post_z.sum(1)

    def predict(self, x, y):

        ypred_logits = self.theta_d(x)
        cross_entropy_loss = F.cross_entropy(ypred_logits, y.argmax(1))
        acc = (ypred_logits.argmax(1) == y.argmax(1)).float().mean()

        yinf_logits = self.phi_y(x)
        inf_cross_entropy_loss = F.cross_entropy(yinf_logits, y.argmax(1))
        inf_acc = (yinf_logits.argmax(1) == y.argmax(1)).float().mean()

        qz_l_param = self.phi_z(torch.cat([x, y], dim=1))
        qz_l_mu = qz_l_param[:, :self.dim_z]
        qz_l_log_sigma_sq = qz_l_param[:, self.dim_z:]
        # Sample from z posterior
        z_l_sample = self._draw_sample(qz_l_mu, qz_l_log_sigma_sq)
        px_l_param = self.theta_g(torch.cat([z_l_sample, y], dim=1))
        px_l_mu = torch.sigmoid(px_l_param)
        ll = utils.bernoulli_logpdf(x, px_l_mu).sum(1).mean()

        return cross_entropy_loss, acc, inf_cross_entropy_loss, inf_acc, ll
