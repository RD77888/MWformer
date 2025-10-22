import logging
import os
import warnings
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from model import MWformer
from utils.loss import WaveletMultiChannelLoss

warnings.filterwarnings('ignore')

model_list = {
            'MWformer': MWformer,
        }

def setup_logger(setting, log_dir='./training_logs'):
    logger = logging.getLogger(f'training_log_{setting}')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{setting}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.logger = None

    def _build_model(self):
        model = model_list[self.args.model]
        model = model.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, learning_rate=None):
        lr = learning_rate if learning_rate is not None else self.args.learning_rate
        return optim.Adam(self.model.parameters(), lr=lr)

    def _select_criterion(self, loss_type=None, level=None):
        loss_type = loss_type if loss_type is not None else self.args.loss
        level_to_use = level if level is not None else getattr(self.args, 'level', 1)

        if loss_type == 'L2':
            return nn.MSELoss()
        elif loss_type == 'L1':
            return nn.L1Loss()
        elif loss_type == 'MTL':
            return WaveletMultiChannelLoss(
                self.args.wavelet_name,
                level_to_use,
                self.args.MTL_loss,
                self.args.delta,
            )
        elif loss_type == 'Smooth_L1':
            return nn.SmoothL1Loss(reduction="mean", beta=self.args.delta)

    def _create_scheduler(self, model_optim, train_loader, train_epochs, learning_rate,
                          pct_start=None, scheduler_type='onecycle'):
        if scheduler_type == 'onecycle':
            pct_start = pct_start if pct_start is not None else getattr(self.args, 'pct_start', 0.2)
            return lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=len(train_loader),
                pct_start=pct_start,
                epochs=train_epochs,
                max_lr=learning_rate,
            )
        elif scheduler_type == 'reduce_on_plateau':
            factor = getattr(self.args, 'lr_factor', 0.9)
            patience = getattr(self.args, 'lr_patience', 3)
            min_lr = getattr(self.args, 'min_lr_rate', 1e-3)
            return lr_scheduler.ReduceLROnPlateau(
                optimizer=model_optim,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=learning_rate*min_lr
            )
        elif scheduler_type == 'cosine_warm_restarts':
            T_0 = getattr(self.args, 'T_0', 10)
            T_mult = getattr(self.args, 'T_mult', 2)
            eta_min = getattr(self.args, 'eta_min', 1e-7)
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=model_optim,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )


    def _log_message(self, message, level='info'):
        if self.logger:
            clean_message = message.replace('\033[94m', '').replace('\033[0m', '')
            getattr(self.logger, level)(clean_message)

    def vali(self, vali_data=None, vali_loader=None, criterion=None):
        if vali_loader is None:
            vali_data, vali_loader = self._get_data(flag='val')

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_runoff = batch_x[0].float().to(self.device)
                batch_rain = batch_x[1].float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, combined_pred = self.model(
                        batch_runoff, batch_rain, batch_x_mark, batch_y, batch_y_mark
                    )

                f_dim = -1
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = combined_pred[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = nn.MSELoss()(pred, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _forward_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_runoff = batch_x[0].float().to(self.device)
        batch_rain = batch_x[1].float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs, combined_pred = self.model(
                batch_runoff, batch_rain, batch_x_mark, dec_inp, batch_y_mark
            )
        f_dim = -1
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, combined_pred, batch_y

    def _compute_loss_and_step(self, outputs, combined_pred, batch_y, model_optim, criterion, loss_type):
        f_dim = -1
        model_optim.zero_grad()
        if loss_type == "MTL":
            loss = criterion(outputs, batch_y)
        else:
            pred_slice = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(pred_slice, batch_y)
        loss.backward()
        model_optim.step()
        return loss

    def _print_epoch_info(self, epoch, train_epochs, avg_train_loss, vali_loss,
                          current_lr, epoch_time, best_train_loss, best_val_loss):
        best_msg = f"\033[94mBest | Train Loss: {best_train_loss:.4e} | Val Loss: {best_val_loss:.4e}\033[0m"
        current_msg = (
            f"Epoch {epoch + 1}/{train_epochs} | Train Loss: {avg_train_loss:.4e} "
            f"| Val Loss: {vali_loss:.4e} | LR: {current_lr:.4e}"
        )
        time_msg = f"Epoch Time: {epoch_time:.2f}s"

        print("\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K", end="", flush=True)
        print(best_msg)
        print(current_msg)
        print(time_msg)
        # 记录到日志
        self._log_message(
            f"Epoch {epoch + 1}/{train_epochs} - Train: {avg_train_loss:.4f}, Val: {vali_loss:.4f}, LR: {current_lr:.6e}, Time: {epoch_time:.2f}s")

    def _check_and_update_best_loss(self, vali_loss, best_val_loss):
        if vali_loss < best_val_loss:
            improvement_msg = f"Validation loss decreased. Saving model ..."
            print(improvement_msg)
            self._log_message(f"val loss descend: {best_val_loss:.6f} -> {vali_loss:.6f}")
            return vali_loss
        return best_val_loss

    def train(self, setting=None):
        pass

    def test(self, setting=None, test=0):
        pass
