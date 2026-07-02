import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer for Fourier Neural Operators.
    Performs FFT, filters/truncates high frequencies, performs complex multiplication,
    and performs Inverse FFT back to the physical domain.
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
        # Explicit complex multiplication using real/imaginary components.
        # This bypasses PyTorch's implicit complex-tensor einsum overhead,
        # resolving the memory stride distribution bottleneck by performing
        # arithmetic operations over contiguous real-valued components.
        # (a + ib) * (c + id) = (ac - bd) + i(bc + ad)
        input_real = input.real
        input_imag = input.imag
        weights_real = weights.real
        weights_imag = weights.imag

        out_real = torch.einsum("bixy,ioxy->boxy", input_real, weights_real) - \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag)
        out_imag = torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) + \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_real)

        # Reconstruct the complex output tensor.
        # Equivalent to: torch.view_as_complex(torch.stack([out_real, out_imag], dim=-1))
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batchsize = x.shape[0]
        # Transform to Fourier domain
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # --- TECHNICAL NOTE ON COMPUTATIONAL STRIDE & MEMORY BANDWIDTH ---
        # Slicing operations like `x_ft[:, :, :self.modes1, :self.modes2]` create non-contiguous 
        # views of the underlying tensors. In CUDA execution layouts, the stride of the tensor 
        # is determined by its dimension sizes. When slicing along the inner spatial dimensions, 
        # the read/write transactions are non-coalesced. 
        #
        # During the real/imaginary component distribution phase of torch.fft.rfft2:
        # 1. Standard PyTorch indexing on sliced views requires gather/scatter memory instructions 
        #    rather than linear bulk copies.
        # 2. This induces out-of-core memory bandwidth limitations, where the memory controller 
        #    spends excessive cycles processing fragmented strides instead of maximizing GPU arithmetic throughput.
        # 3. For large models or grid resolutions, pre-allocating contiguous blocks or using custom 
        #    fused CUDA kernels for spectral slicing is recommended to alleviate this bottleneck.
        # ------------------------------------------------------------------
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        # Transform back to physical domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
