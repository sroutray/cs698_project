import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class M2(nn.Module):

    def __init__(   self,
                    device, gamma,
                    num_batches, batch_size,
                    dim_x=28*28, dim_z=100, dim_y=10):
        super(M2, self).__init__()

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

        self.theta_g = nn.Sequential(nn.Linear(dim_z+dim_y, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, 500), nn.ReLU(inplace=True),
                                     nn.Linear(500, dim_x))
                                    
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
        ypred_l_logits = self.phi_y(x_l)
        D_l = - F.cross_entropy(ypred_l_logits, y_l.argmax(1), reduction='none')
        D_l = D_l.sum()   
        
        
        
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
        L_tot = L_l + self.gamma * D_l + L_u
        loss = - L_tot / self.batch_size


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

    def predict(self, x, y, compute_ml = False):

        ypred_logits = self.phi_y(x)
        cross_entropy_loss = F.cross_entropy(ypred_logits, y.argmax(1))
        acc = (ypred_logits.argmax(1) == y.argmax(1)).float().mean()

        if compute_ml:
            
            S = 1
            ypred_logits = self.phi_y(x)
            ypred = F.softmax(ypred_logits, 1)
            
            ypredsamples = torch.multinomial(ypred, S , replacement=True)

            ml = 0
            
            for i in range(S):
                y_sample = F.one_hot(ypredsamples[:,i].reshape(-1,), num_classes = self.dim_y)
                y_sample = torch.FloatTensor(y_sample)
                
                #distribution for q_z
                qz_param = self.phi_z(torch.cat([x, y_sample], dim=1))
                qz_mu = qz_param[:, :self.dim_z]
                qz_log_sigma_sq = qz_param[:, self.dim_z:]
                
                for _ in range(S):

                    z_sample = self._draw_sample(qz_mu, qz_log_sigma_sq)
                    #distribution p(x|y,z)
                    px_param = self.theta_g(torch.cat([z_sample, y_sample], dim=1))
                    px_mu = torch.sigmoid(px_param)
                    
                    
                    log_lik =  utils.bernoulli_logpdf(x, px_mu).sum()
                    log_prior_z = utils.stdnormal_logpdf(z_sample).sum()
                    prior_y = 1/self.dim_y

                    log_posterior_z = utils.normal_logpdf(z_sample, qz_mu, qz_l_log_sigma_sq).sum()
                    log_posterior_y = F.cross_entropy(ypred, ypredsamples, reduction='none').sum()
                    
                    ml += prior_y*torch.exp(log_lik +  log_prior_z - log_posterior_z - log_posterior_y)
                


        if not compute_ml:
            return cross_entropy_loss, acc
        else:
            return cross_entropy_loss, acc, ml