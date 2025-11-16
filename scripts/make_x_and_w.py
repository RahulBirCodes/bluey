import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import MultiHeadAttention
from optimizers.muonW1 import MuonW
from optimizers.manifold_muonW import ManifoldMuonW
from model.model import Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def make_xy_payloads(batch_size, num_pairs, mu=0.0, sigma=1.0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # x: (B, T, 5)
    x = mu + sigma * torch.randn(batch_size, num_pairs, 5, device=device)

    # Task-specific linear map W, b per batch element (or per batch if you want shared task)
    W = torch.randn(batch_size, 1, 5, 5, device=device)  # (B, 1, 5, 5)
    b = torch.randn(batch_size, 1, 1, device=device)  # (B, 1, 1)

    # y: scalar per pair, or you can make it 5-d, your choice
    y_scalar = (x * W).sum(-1, keepdim=True) + b        # (B, T, 1)
    
    # Optionally project to 5-d
    y = torch.cat([y_scalar, torch.zeros_like(x[..., 1:])], dim=-1)  # (B, T, 5)



    return x, y, W, b


def get_dataset(dataset_size=20, seq_len=5, d_model=12) -> tuple:
    """
    We are making synthetic data for an autoregressive decoder-only 10 million parameter transformer model
    to train and validate on. The model should look at batches of sequences of input vectors 12 floats long,
    with each input vector alternately being an x or a y. To keep the affine subspaces of the embedding and
    Q, K, V separate, we want to create our input vectors in a way such that the first two floats are flags for
    x and y respectively and the next 10 floats are representing five floats of x or five floats of y.

    Ideally the embedder should learn some process by which to encode these vectors differently based on the flags. 
    When creating the data, we want the RMS-norm of the input vector, to be equal to one, with each float
    randomly distributed, maybe normally, according to some parameters mu and sigma, which the user should be able to
    specify. 

    The function should ideally create a bunch of x's and then a bunch of y's, which can then by returned, and interspersed
    together so that some sequences start with x's and some sequences start with y's, to prevent beginning bias.

    What's the best way to do this? The task that we're replicating comes from this article: 
    https://arxiv.org/pdf/2208.01066 with this repo: https://github.com/dtsip/in-context-learning/tree/main/src
    
    I guess it can return a tuple of data? Or should it return the full list? 
    """
    #Needs to be [(one digit flagging x) (one digit flagging y) (5 digits of x) (5 digits y)]

    #I think that we want the 5 digits of x or y to be randomly distributed with RMS norm 1, so that activations
    #are consistent

    #I think that we only want the x digits to be on when the x flag is on, when x flag is on, all the y
    #flags and other things should be 0. Vice verse if the y flag is on.
    x = torch.randn(dataset_size, seq_len, 5, device=device)
    y = torch.randn(dataset_size, seq_len, 5, device=device)

    #Make ones and zeros of dataset_size / 2
    ones = torch.ones((dataset_size/2, seq_len, 1))
    zeros = torch.zeros((dataset_size/2, seq_len, 1))

    #Alternates ones and zeros of dataset_size / 2
    x_flag = torch.cat((ones, zeros), dim=2) #half
    y_flag = torch.cat((zeros, ones), dim=2) #half

    #Make (dataset_size, seq_len, 10)
    ten = torch.cat((x, y), dim=2)

    #Cut into halves: (dataset_size/2, seq_len, 10) x first half, y second half
    x_half = ten[:dataset_size/2, :, :]
    y_half = ten[dataset_size/2:, :, :]
    
    #Add flags: (dataset_size/2, seq_len, 2) to halves: (dataset_size/2, seq_len, 10) 
    ten_x = torch.cat((x_flag, x_half), dim=2)
    ten_y = torch.cat((y_flag, y_half), dim=2)

    #Add both x: (dataset_size/2, seq_len, 12) to y: (dataset_size/2, seq_len, 12)
    full = torch.cat((ten_x, ten_y), dim=0)

    return full


def get_batch(batch_size=8, seq_len=5, d_model=12):
    """
    We are making synthetic data for an autoregressive decoder-only 10 million parameter transformer model
    to train and validate on. The model should look at batches of sequences of input vectors 12 floats long,
    with each input vector alternately being an x or a y. To keep the affine subspaces of the embedding and
    Q, K, V separate, we want to create our input vectors in a way such that the first two floats are flags for
    x and y respectively and the next 10 floats are representing five floats of x or five floats of y.

    Ideally the embedder should learn some process by which to encode these vectors differently based on the flags. 
    When creating the data, we want the RMS-norm of the input vector, to be equal to one, with each float
    randomly distributed, maybe normally, according to some parameters mu and sigma, which the user should be able to
    specify. 

    The function should ideally create a bunch of x's and then a bunch of y's, which can then by returned, and interspersed
    together so that some sequences start with x's and some sequences start with y's, to prevent beginning bias.

    What's the best way to do this? The task that we're replicating comes from this article: 
    https://arxiv.org/pdf/2208.01066 with this repo: https://github.com/dtsip/in-context-learning/tree/main/src
    
    """
    #Needs to be [(one digit flagging x) (one digit flagging y) (5 digits of x) (5 digits y)]

    #I think that we want the 5 digits of x or y to be randomly distributed with RMS norm 1, so that activations
    #are consistent

    #I think that we only want the x digits to be on when the x flag is on, when x flag is on, all the y
    #flags and other things should be 0. Vice verse if the y flag is on.


    return torch.randn(batch_size, seq_len, d_model, device=device)

