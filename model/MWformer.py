import torch
import torch.nn as nn
from pytorch_wavelets import IDWT1D

from layers.DWT import Decomposition, calculate_decomposed_dims
from layers.DualAttn import StackDualAttention
from layers.Patch import TFPE
from layers.Pred import MWPD

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input = configs.num_runoff+configs.num_rain
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.level = configs.level
        self.wavelet_name = configs.wavelet_name
        self.batch_size = configs.batch_size
        self.n_heads = configs.n_heads
        self.d_ff=configs.d_ff
        self.e_layers = configs.e_layers
        self.patch_num = configs.patch_num
        self.dropout = configs.dropout

        self.seq_lens =calculate_decomposed_dims(self.seq_len,self.level,self.wavelet_name)
        self.pred_lens = calculate_decomposed_dims(self.pred_len,self.level, self.wavelet_name)
        self.decomposition = Decomposition(
            input_length=self.seq_len,
            wavelet_name=self.wavelet_name,
            level=self.level,
            channel=self.input,
            d_model=self.d_model,
        )
        self.embed = nn.ModuleList(
            [
                TFPE(
                    target_patch_num=self.patch_num,
                    band_len=self.seq_lens[i],
                    input_channels=self.input ,
                    d_model=self.d_model,
                )for i in range(self.level+1)
            ]
        )

        self.DualAttention = StackDualAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_layers=self.e_layers,
        )

        self.predictors = MWPD(
            patch_num=self.patch_num,
            d_model=self.d_model,
            pred_lens=self.pred_lens,
            d_ff=configs.d_ff,
            dropout=configs.dropout
        )

        self.freq_embeddings = nn.Parameter(
            torch.randn(1, 1, self.level+1, self.d_model) * 0.02
        )
        self.idwt = IDWT1D(wave=self.wavelet_name, mode='symmetric')

    def forward(self, runoff,rainfall,batch_x_mark=None, batch_y=None, batch_y_mark=None):

        x_input= torch.cat((runoff, rainfall), dim=-1)
        # Reshape [batch, num_runoff, seq_len]
        x_decomposed_input = x_input.permute(0, 2, 1)

        # 2. Wavelet Decomposition
        # list [cA, cD_J, ..., cD_1] from calculate_decomposed_dims
        decomposed_bands = self.decomposition(x_decomposed_input)
        # 3. Patch Embedding for each band
        embedded_bands = []
        for i in range(self.level + 1):
            embedded_output = self.embed[i](decomposed_bands[i])
            embedded_bands.append(embedded_output)

        stacked_features = torch.stack(embedded_bands, dim=2)
        stacked_features = stacked_features + self.freq_embeddings

        # 5. Pass through StackedDualAttention
        #current_features shape remains: [batch, patch_num, level+1, d_model]
        current_features = self.DualAttention(stacked_features)
        # 6. Add frequency embeddings


        # 6. Prediction for each frequency band
        wavelet_channel_predictions=self.predictors(current_features)

        cA_pred_weighted = wavelet_channel_predictions[0]
        cDs_pred_weighted = []
        for i in range(len(wavelet_channel_predictions) - 1):  # Iterate for cD components
            # Weight index for cDs starts from freq_weights[1]
            cDs_pred_weighted.append(wavelet_channel_predictions[i + 1])

        # 8. Inverse Wavelet Transform
        combined_output = self.idwt((cA_pred_weighted, cDs_pred_weighted))
        final_prediction = combined_output.permute(0, 2, 1)
        return wavelet_channel_predictions,final_prediction
