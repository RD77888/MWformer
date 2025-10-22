import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D

class Decomposition(nn.Module):
    def __init__(self,
                 input_length,
                 wavelet_name='db1',
                 level=1,
                 channel=1,
                 d_model=128,
                 padding_mode='symmetric',
                 ):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.channel_init_arg = channel
        self.output_length = d_model
        self.padding_mode = padding_mode

        self.dwt = DWT1D(wave=self.wavelet_name, J=self.level, mode=self.padding_mode)
        self.decomposed_coeff_lengths = self._calculate_decomposed_dims(self.input_length)

    def forward(self, x):
        return self.transform(x)

    def transform(self, x):
        yl, yh_list = self.dwt(x)
        coeff_list = [yl] + [yh_list[i] for i in range(self.level)]
        return coeff_list

    def _calculate_decomposed_dims(self, current_input_length):
        dummy_x_cpu = torch.ones((1, 1, current_input_length))
        temp_dwt_for_shape = DWT1D(wave=self.wavelet_name, J=self.level, mode=self.padding_mode)
        yl, yh = temp_dwt_for_shape(dummy_x_cpu)
        dimensions = [yl.shape[-1]]  # cA
        for cD in yh:
            dimensions.append(cD.shape[-1])
        return dimensions


def calculate_decomposed_dims(pred_len: int,
                              level: int,
                              wavelet_name: str,
                              padding_mode: str = 'symmetric') -> list[int]:

    dwt_temp = DWT1D(wave=wavelet_name, J=level, mode=padding_mode)
    dummy_input = torch.randn(1, 1, pred_len)
    cA_dummy, cDs_dummy_list = dwt_temp(dummy_input)
    coeff_lengths = [cA_dummy.shape[-1]]
    for cD_dummy in cDs_dummy_list:
        coeff_lengths.append(cD_dummy.shape[-1])
    return coeff_lengths


