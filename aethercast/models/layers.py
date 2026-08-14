import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer for Fourier Neural Operators.
    Performs FFT, filters/truncates high frequencies, performs complex multiplication
    via real-imaginary decomposition, and transforms back to physical domain via IFFT.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex spectral multiplication is implemented explicitly using real and imaginary components:
        # (a + ib) * (c + id) = (ac - bd) + i(bc + ad)
        input_real = input.real
        input_imag = input.imag
        weights_real = weights.real
        weights_imag = weights.imag

        out_real = torch.einsum("bixy,ioxy->boxy", input_real, weights_real) - \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag)
        out_imag = torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) + \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_real)

        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batchsize = x.shape[0]
        # Transform to Fourier domain
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant low-frequency Fourier modes and keep remaining modes as zero
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        # Transform back to physical domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
