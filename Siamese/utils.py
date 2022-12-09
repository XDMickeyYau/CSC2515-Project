import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(anchor, positive, negative):
    """
    shows an imagenet-normalized image on the screen
    """

    anchor = anchor.numpy().squeeze()
    positive = positive.numpy().squeeze()
    negative = negative.numpy().squeeze()

    # # show image
    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(np.transpose(anchor, (1, 2, 0)))
    ax.set_title("Anchor")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(np.transpose(positive, (1, 2, 0)))
    ax.set_title("Positive")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(np.transpose(negative, (1, 2, 0)))
    ax.set_title("Negative")

    plt.show()


def triplet_loss(y_true, y_pred):
    alpha = 0.6

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pd = torch.sum(torch.square(anchor - positive), axis=1)
    nd = torch.sum(torch.square(anchor - negative), axis=1)
    zero = torch.zeros_like(pd) 

    return torch.maximum(pd - nd + alpha, zero)