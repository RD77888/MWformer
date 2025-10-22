import os
import time

import numpy as np
import torch

from exp.exp_basic import Exp_Basic, setup_logger
from utils.metrics import metric
from utils.tools import EarlyStopping


class Exp_main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def train(self, setting):
        self.logger = setup_logger(setting)
        self._log_message(f"start,setup: {setting}")

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        print("")
        print("")
        print("")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion().to(self.device)
        scheduler = self._create_scheduler(
            model_optim, train_loader, self.args.train_epochs, self.args.learning_rate,
            scheduler_type=self.args.scheduler
        )
        best_train_loss = float("inf")
        best_val_loss = float("inf")

        self._log_message(f"train - Epochs: {self.args.train_epochs}, learning rate: {self.args.learning_rate}")

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_train_loss = []
            start_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                outputs, combined_pred, batch_y = self._forward_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                loss = self._compute_loss_and_step(
                    outputs, combined_pred, batch_y, model_optim, criterion, self.args.loss
                )
                epoch_train_loss.append(loss.item())

                if self.args.scheduler == 'onecycle':
                    scheduler.step()

            avg_train_loss = np.mean(epoch_train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss

            if self.args.scheduler == 'reduce_on_plateau':
                scheduler.step(vali_loss)

            current_lr = scheduler.get_last_lr()[0]
            epoch_time = time.time() - start_time

            self._print_epoch_info(
                epoch, self.args.train_epochs, avg_train_loss, vali_loss,
                current_lr, epoch_time, best_train_loss, best_val_loss,
            )

            best_val_loss = self._check_and_update_best_loss(vali_loss, best_val_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                self._log_message(f"Early stopping at epoch  {epoch + 1} ")
                break
        self._log_message(f"train completed - best val loss: {best_val_loss:.6f}")
        return self.model

    def test(self, setting, test=0):
        """测试方法"""
        self.logger = setup_logger(f"{setting}_test")
        self._log_message(f"start,setup: {setting}")

        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            checkpoint_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self._log_message(f"load model: {checkpoint_path}")

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                outputs, combined_pred, batch_y = self._forward_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                f_dim = -1
                pred = combined_pred[:, -self.args.pred_len:, f_dim:]

                pred = pred.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        result_msg = f"mse:{mse:.6f}, mae:{mae:.6f}, rse:{rse:.6f}"
        print(result_msg)
        self._log_message(f"result - {result_msg}")

        with open("result.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(result_msg + "\n")

        return mae, mse, rmse, mape, mspe, rse, corr
