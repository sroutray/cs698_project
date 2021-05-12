import torch
import numpy as np
import sys

logc = np.log(2.*np.pi)
c = - 0.5 * np.log(2*np.pi)

def normal_logpdf(x, mu, log_sigma_sq):

	return ( - 0.5 * logc - log_sigma_sq / 2. - torch.div( (x - mu)**2, 2 * torch.exp( log_sigma_sq ) ) )

def stdnormal_logpdf(x):

	return ( - 0.5 * ( logc + x**2 ) )

def bernoulli_logpdf(x, mu, const=1e-10):

	return ( x * torch.log( torch.clamp(mu, const, 1.0) ) + \
			(1.0 - x) * torch.log( torch.clamp(1.0 - mu, const, 1.0) ) )

def gaussian_ent(log_sigma_sq):

	return ( - 0.5 * ( logc + 1.0 + log_sigma_sq ) )

def gaussian_marg(mu, log_sigma_sq):

	return ( - 0.5 * ( logc + ( mu**2  + torch.exp( log_sigma_sq ) ) ) )

# def tf_binary_xentropy(x, y, const = 1e-10):

#     return - ( x * tf.log ( tf.clip_by_value( y, const, 1.0 ) ) + \
#              (1.0 - x) * tf.log( tf.clip_by_value( 1.0 - y, const, 1.0 ) ) )

# def feed_numpy_semisupervised(num_lab_batch, num_ulab_batch, x_lab, y, x_ulab):

# 	size = x_lab.shape[0] + x_ulab.shape[0]
# 	batch_size = num_lab_batch + num_ulab_batch
# 	count = int(size / batch_size)

# 	dim = x_lab.shape[1]

# 	for i in xrange(count):
# 		start_lab = i * num_lab_batch
# 		end_lab = start_lab + num_lab_batch
# 		start_ulab = i * num_ulab_batch
# 		end_ulab = start_ulab + num_ulab_batch

# 		yield [	x_lab[start_lab:end_lab,:dim/2], x_lab[start_lab:end_lab,dim/2:dim], y[start_lab:end_lab],
# 				x_ulab[start_ulab:end_ulab,:dim/2], x_ulab[start_ulab:end_ulab,dim/2:dim] ]

# def feed_numpy(batch_size, x):

# 	size = x.shape[0]
# 	count = int(size / batch_size)

# 	dim = x.shape[1]

# 	for i in xrange(count):
# 		start = i * batch_size
# 		end = start + batch_size

# 		yield x[start:end]

def print_metrics(epoch, *metrics):

	print(25*'-')
	for metric in metrics: 
		print('[{}] {} {}: {}'.format(epoch, metric[0],metric[1],metric[2]))
	sys.stdout.flush()
	print(25*'-')