import torch
import torch.nn as nn
import numpy as np


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        ## Choose one
        # We compute the interpolated values of the various functions defined by their Fourier coefficients at each input coordinate and sum them.
        # y =  torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        # y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        # if self.addbias:
        #     y += self.bias
        # #End fuse

        # You can use einsum to reduce memory usage. It's not as good as full fusion, but it should help. Einsum is usually slower though.
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias

        # Debug: Print shapes to ensure correctness
        # print(f"Shape of x: {xshp}")
        # print(f"Shape of y before view: {y.shape}")
        # print(f"Expected outshape: {outshape}")

        # Ensure total number of elements matches
        if y.numel() != np.prod(outshape):
            raise RuntimeError(f"Shape {outshape} is invalid for input of size {y.numel()}")

        y = y.view(outshape)
        return y
