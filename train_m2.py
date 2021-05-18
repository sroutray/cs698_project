from datetime import datetime
import numpy as np
import data.mnist as mnist
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim
import torch.nn as nn
import utils
from m2 import M2


class MNIST(Dataset):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __len__(self):

        return self.y.shape[0]

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        return torch.Tensor(x), torch.Tensor(y)


def train(  model, optimizer, device,
            loader_ulab, loader_lab):
    
    num_iters = len(loader_ulab)
    train_loss = 0
    loader_ulab = iter(loader_ulab)
    loader_lab = iter(loader_lab)

    for its in range(num_iters):

        x_u, y_u = loader_ulab.next()
        x_l, y_l = loader_lab.next()
        x_u, x_l, y_l = x_u.to(device), x_l.to(device), y_l.to(device)
        
        loss = model(x_l, y_l, x_u)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / num_iters


def evaluate(model, device, loader_valid, loader_train = None):

    num_iters = len(loader_valid)
    acc = 0
    loss = 0
    loader_valid = iter(loader_valid)

    for its in range(num_iters):
        
        x, y = loader_valid.next()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            loss_its, acc_its = model.predict(x, y)
        
        loss += loss_its.item()
        acc += acc_its.item()

    #compute marginal likelihood if loader_train is passed
    if loader_train:
        loader_train = iter(loader_train)
        num_iters = len(loader_train)
        marginal_likelihood = 0
        for its in range(num_iters):
            x, y = loader_train.next()
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                loss_its, acc_its, ml = model.predict(x, y, compute_ml = True)

            marginal_likelihood += ml.item()
    
    if not loader_train:
        return loss / num_iters, acc / num_iters
    else:
        return loss / num_iters, acc / num_iters, marginal_likelihood


if __name__ == '__main__':

    ## Experiment Parameters ##

    num_lab = 100           #Number of labelled examples (total)
    num_batches = 100       #Number of minibatches in a single epoch
    dim_z = 100              #Dimensionality of latent variable (z)
    epochs = 1001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    weight_dis = 5.0             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
    num_workers = 4
    stop_iter = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_every = 1
    chkpt_str = datetime.now().strftime('%b%d_%H-%M-%S')

    ## Load Dataset ##

    mnist_path = 'mnist/mnist_28.pkl.gz'
    #Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(mnist_path, binarize_y=True) 
    x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, num_lab)
    x_lab, y_lab = x_l.T, y_l.T
    x_ulab, y_ulab = x_u.T, y_u.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T
    print("Unlabelled:", x_ulab.shape, y_ulab.shape)
    print("Labelled:", x_lab.shape, y_lab.shape)
    print("Validation:", x_valid.shape, y_valid.shape)
    print("Test:", x_test.shape, y_test.shape)


    ## Setup Experiment ##

    num_examples = x_lab.shape[0] + x_ulab.shape[0]
    num_ulab = num_examples - num_lab

    assert num_lab % num_batches == 0, '#Labelled % #Batches != 0'
    assert num_ulab % num_batches == 0, '#Unlabelled % #Batches != 0'
    assert num_examples % num_batches == 0, '#Examples % #Batches != 0'

    batch_size = num_examples // num_batches
    num_lab_batch = num_lab // num_batches
    num_ulab_batch = num_ulab // num_batches

    gamma = weight_dis * batch_size / num_lab_batch
    

    ## Create Dataloaders ##

    data_ulab = MNIST(x_ulab, y_ulab)
    data_lab = MNIST(x_lab, y_lab)
    data_valid = MNIST(x_valid, y_valid)
    data_test = MNIST(x_test, y_test)

    loader_ulab = DataLoader(   data_ulab, batch_size=num_ulab_batch,
                                shuffle=True, drop_last=True, num_workers=num_workers)
    loader_lab = DataLoader(    data_lab, batch_size=num_lab_batch,
                                shuffle=True, drop_last=True, num_workers=num_workers)
    loader_valid = DataLoader(  data_valid, batch_size=batch_size,
                                shuffle=False, drop_last=False, num_workers=num_workers)
    loader_test = DataLoader(   data_test, batch_size=batch_size,
                                shuffle=False, drop_last=False, num_workers=num_workers)
    assert len(loader_ulab) == len(loader_lab)


    ## Create Model ##

    model = M2( device, gamma,
                num_batches, batch_size,
                dim_x=28*28, dim_z=dim_z, dim_y=10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    ## Training ##

    best_valid_acc = 0.
    stop_iter_count = 0

    for epoch in range(epochs):
   
        train_loss = train( model, optimizer, device,
                            loader_ulab, loader_lab)

        if np.isnan(train_loss):
            chkpt = torch.load('./checkpoints/best_val_acc.pt')
            model.load_state_dict(chkpt['model'])
            optimizer.load_state_dict(chkpt['optimizer'])

        model.eval()
        valid_loss, valid_acc, marginal_likelihood = evaluate(model, device, loader_valid, loader_lab)
        model.train()

        # stop_iter_count += 1
        # if stop_iter_count > stop_iter:
        #     break

        if best_valid_acc <= valid_acc:
            stop_iter_count = 0
            best_valid_acc = valid_acc
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        './checkpoints/best_val_acc_'+chkpt_str+'.pt') 
        
        if epoch % print_every == 0:
            utils.print_metrics( 	epoch,
                                    ['Training', 'cost', train_loss],
                                    ['Validation', 'accuracy', valid_acc],
                                    ['Validation', 'cross-entropy', valid_loss],
                                    ['Training', 'Marginal Likelihood', marginal_likelihood] )       

            
    