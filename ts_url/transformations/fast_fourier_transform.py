from ..registry import TRANSFORMATION
from .ts_transformation import VoidTransfromation
import torch
__all__ = ["FFT"]

@TRANSFORMATION.register("fft")
class FFT(VoidTransfromation):
    def transform(self, x, index, **kwargs):
        # Compute complex-to-complex FFT
        fft_result = torch.fft.fft(x)

        # Split complex tensor into real and imaginary parts
        # real_parts = fft_result.real
        # imaginary_parts = fft_result.imag
        fft_result = torch.abs(fft_result)

        # Concatenate real and imaginary parts along a new dimension
        # fft_input = torch.cat((real_parts, imaginary_parts), dim=-1)
        return fft_result