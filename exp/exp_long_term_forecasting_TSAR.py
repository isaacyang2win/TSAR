from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast_TSAR(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_TSAR, self).__init__(args)
        self.lambda_firstscale = args.lambda_48
        self.scale_num = int(math.ceil(math.log2((args.pred_len + args.label_len) / args.token_len)) + 1)
        print(f"scale_num: {self.scale_num}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, data=None):
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss_per_scale = [[] for _ in range(self.scale_num)]
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, *_ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                label_y, outputs = self.model.ar_trainning(batch_x, batch_y)
                for si, (label, out) in enumerate(zip(label_y, outputs)):
                    loss = criterion(out.detach().cpu(), label.detach().cpu())
                    total_loss_per_scale[si].append(loss.item())
        averaged_losses = [np.average(scale_loss) for scale_loss in total_loss_per_scale]
        weights = [self.lambda_firstscale] + [1.0] * (self.scale_num - 1)
        total_weight = sum(weights)
        final_val_loss = sum(w * l for w, l in zip(weights, averaged_losses)) / total_weight
        self.model.train()
        return final_val_loss

    def train(self, setting):
        train_loss_per_scale = [[] for _ in range(self.scale_num)]
        weights = [self.lambda_firstscale] + [1.0] * (self.scale_num - 1)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        iter_verbose = 1000
        if self.args.load_pretrain:
            iter_verbose = 100
            print('loading')
            setting2 = setting.replace(self.args.data, 'pretrain')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting2, 'checkpoint.pth'), map_location=self.device))
            param_train = 0
            param_all = 0
            for name, param in self.model.named_parameters():
                if 'forecast_head' in name:
                    print(name)
                    param_train += param.numel()
                    param_all += param.numel()
                else:
                    param.requires_grad = False
                    param_all += param.numel()
            print(
                f'trainable parameters num: {clever_format(param_train)}, all parameters num: {clever_format(param_all)},'
                f'ratio: {param_train / param_all * 100} %')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_per_scale = [[] for _ in range(self.scale_num)]
            total_grad_norm = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                label_y, outputs = self.model.ar_trainning(batch_x, batch_y)
                losses = [criterion(out, label) for label, out in zip(label_y, outputs)]
                for si, l in enumerate(losses):
                    train_loss_per_scale[si].append(l.item())
                total_weight = sum(weights)
                loss = sum(w * l for w, l in zip(weights, losses)) / total_weight

                if (i + 1) % iter_verbose == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                total_grad_norm.append(total_norm.item())
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            averaged_train_losses = [np.average(scale_loss) for scale_loss in train_loss_per_scale]
            train_loss = sum(w * l for w, l in zip(weights, averaged_train_losses)) / total_weight
            avg_grad_norm = np.mean(total_grad_norm)
            print(f"Epoch {epoch+1} average grad norm: {avg_grad_norm:.4f}")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                pred_tmp = []
                batch_x_backup = batch_x
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.ar_pred_len:, :].to(self.device)
                batch_y2 = batch_y.detach().cpu().numpy()
                batch_y2 = batch_y2[:, :, f_dim:]
                true = batch_y2
                batch_x = batch_x.float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model.autoregressive_infer_cfg(batch_x)
                else:
                    outputs = self.model.autoregressive_infer_cfg(batch_x)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                pred_tmp.append(outputs[:, :, f_dim:])

                pred = torch.cat(pred_tmp, dim=1).detach().cpu().numpy()[:, :self.args.ar_pred_len, :]
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_np = batch_x_backup.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_batch = total_time / len(test_loader)

        print(f"\nTest Time Summary:")
        print(f"  Total test time: {total_time:.2f} seconds")
        print(f"  Average time per batch: {avg_time_per_batch:.4f} seconds")
        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return mae, mse
