import torch
from torch.nn import functional as F
from torch import nn

import math

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class MDC(nn.Module):
    def __init__(self, configs):
        super(MDC, self).__init__()
        self.seq_len = configs.seq_len
        k = 2
        self.k =k
        c =2
        layernorm = True
        c_in = configs.enc_in
        self.d_model = configs.d_model
        self.token_len = 24
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
        self.layernorm = layernorm
        self.c_in = c_in
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.seq_len * self.c_in)

        self.patch_embedding =PatchEmbedding(self.d_model, self.token_len, self.token_len, 0, configs.dropout)
        self.sos_embedding = nn.Linear(self.token_len, self.d_model, bias=False)

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        sample_x = []
        for i, k in enumerate(self.k_list):
            down_x = self.avg_pools[i](x)
            sample_x.append(down_x)
        sample_x.append(x)
        sample_all_scale = torch.cat(sample_x, dim=2)
        sample_all_scale = self.patch_embedding.patch_reshape(sample_all_scale)
        prefix_tokens = self.sos_embedding(sample_all_scale)
        return prefix_tokens

class MDM(nn.Module):
    def __init__(self, seq_len, configs, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = seq_len
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        self.enc_in = configs.enc_in
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.seq_len * self.enc_in)

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        return sample_x[n - 1]

class TSARBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.token_len
        d_model = configs.d_model
        d_ff = configs.d_ff
        self.patch_embedding = PatchEmbedding(d_model, self.patch_size, self.patch_size, 0, configs.dropout)
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model,
                        configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),                    
                    d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.forecast_head = nn.Linear(d_model, configs.token_len)
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.backbone = TSARBlock(configs)
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.forecast_head
        self.enc_embedding = self.backbone.patch_embedding
        self.output_attention = configs.output_attention

        self.pred_len = configs.pred_len + configs.label_len
        self.token_len = configs.token_len
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.first_emb = nn.Linear(self.seq_len, self.token_len)

        self.value_embedding = nn.Linear(self.token_len, self.d_model, bias=False)
        self.sos_embedding = nn.Linear(self.token_len, self.d_model, bias=False)

        self.cond_embedding = nn.Linear(self.token_len, self.d_model, bias=False)

        self.patch_nums = self.calculate_patch_nums(self.pred_len, self.token_len)
        self.L = sum(pn for pn in self.patch_nums)
        self.first_l = 3
        self.context_len = self.first_l
        self.SN = len(self.patch_nums)
        self.pastmixing = MDM(self.seq_len, configs=configs, k=configs.k,  c=2, layernorm=True)

        self.cond_mdc = MDC(configs)
        init_std = math.sqrt(1 / self.d_model / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.d_model))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            if i > 0:
                pe = torch.empty(1, pn, self.d_model)
            else:
                pe = torch.empty(1, self.context_len, self.d_model)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)
        assert tuple(pos_1LC.shape) == (1, self.L + self.context_len - 1, self.d_model)
        self.pos_1LC = nn.Parameter(pos_1LC)

        self.lvl_embed = nn.Embedding(self.SN, self.d_model)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        d: torch.Tensor = torch.cat(
            [torch.full((self.context_len,), 0)]
            + [
                torch.full((pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + self.context_len - 1, 1)

        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(
            1, 1, self.L + self.context_len -1, self.L + self.context_len -1)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        self.first_token = nn.Parameter(torch.randn(1, 1, self.d_model))

    def calculate_patch_nums(self, pred_len, token_len):
        num_tokens = pred_len // token_len
        max_k = int(math.floor(math.log2(num_tokens)))
        scales = [2**k for k in range(max_k + 1)]
        if scales[-1] != num_tokens:
            scales.append(num_tokens)
        return scales


    def ar_trainning(self, x_enc, y_true):
        ed = self.L + self.context_len - 1

        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        y_true = (y_true - means) / stdev

        sos = self.pastmixing(x_enc.permute(0, 2, 1))
        sos = self.enc_embedding.patch_reshape(sos)
        first_token = self.first_token.repeat((sos.shape[0], 1, 1))

        cond_tokens = self.cond_mdc(x_enc.permute(0, 2, 1))

        SN = self.SN

        f_hat =  y_true.new_zeros(y_true.shape).permute(0, 2, 1)

        f_rest = y_true.clone().permute(0, 2, 1)

        next_scales  = []

        all_res = []


        for si in range(SN-1):
            r = F.interpolate(f_rest, size=self.patch_nums[si]*self.token_len, mode='linear', align_corners=True)
            all_res.append(r)
            z = F.interpolate(r, size=self.pred_len, mode='linear', align_corners=True)
            f_hat += z
            h_next = F.interpolate(f_hat, size=self.patch_nums[si+1]*self.token_len, mode='linear', align_corners=True)
            h_next_token = self.enc_embedding.patch_reshape(h_next)
            next_scales.append(h_next_token)
            f_rest = f_rest - z
        
        all_res.append(f_rest)

        
        f_wo_first = torch.cat(next_scales, dim=1)

        sos = self.sos_embedding(sos)
        f_wo_first = self.value_embedding(f_wo_first)
        sos = torch.cat([sos, first_token], dim=1)
        sos = sos + self.pos_start.expand(B*M, self.first_l,  -1)


        AR_input = torch.cat([sos, f_wo_first], dim=1)
        AR_input += (self.lvl_embed(self.lvl_1L[:, :ed].expand(B*M, -1)) + self.pos_1LC[:, :ed])
            
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        dec_out, attns = self.decoder(AR_input,cond_tokens,attn_bias)
        dec_out = self.proj(dec_out)


        outputs = []
        f_scales = []
        start_idx = self.first_l - 1
        for i, patch_num in enumerate(self.patch_nums):
            end_idx = start_idx + patch_num
            out = dec_out[:, start_idx:end_idx, :self.token_len].reshape(B, M, -1).transpose(1, 2)
            out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
            outputs.append(out)

            
            r = all_res[i]
            f_scale = r.permute(0, 2, 1) * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
            f_scales.append(f_scale)

            start_idx = end_idx

        return tuple(f_scales), tuple(outputs)

    @torch.no_grad()
    def autoregressive_infer_cfg(self, x_enc):
        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        SN = self.SN

        for attn_layer in self.decoder.attn_layers:
            attn_layer.self_attention.inner_attention.kv_caching(True)

        sos = self.pastmixing(x_enc.permute(0, 2, 1))
        sos = self.enc_embedding.patch_reshape(sos)

        cond_tokens = self.cond_mdc(x_enc.permute(0, 2, 1))

        first_token = self.first_token.repeat((sos.shape[0], 1, 1))

        ps = self.pos_start.expand(B*M, self.first_l, -1)
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        sos = self.sos_embedding(sos) 
        sos = torch.cat([sos, first_token], dim=1)
        next_token_map = sos + ps + lvl_pos[:, : self.first_l]


        f_hat = torch.zeros(B * M, 1, self.pred_len, device=x_enc.device)
        cur_L = 0

        for si in range(SN):
            cur_L += self.patch_nums[si]

            x = next_token_map

            dec_out, attns = self.decoder(x, cond_tokens)
            dec_out = self.proj(dec_out)

            if si == 0:
                dec_out = dec_out[:, self.first_l - 1:, :]
            r_pred = dec_out.reshape(B * M, 1, -1)

            z_pred = F.interpolate(r_pred, size=self.pred_len, mode='linear', align_corners=True)

            f_hat = f_hat + z_pred

            if si < SN-1:
                h_next = F.interpolate(f_hat, size=self.patch_nums[si + 1] * self.token_len, mode='linear', align_corners=True)
                next_token_map = self.enc_embedding.patch_reshape(h_next)
                next_token_map = self.value_embedding(next_token_map)
                next_token_map += lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1]]

        final_rec = f_hat.reshape(B, M, -1).permute(0, 2, 1)
        final_rec = final_rec * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        for attn_layer in self.decoder.attn_layers:
            attn_layer.self_attention.inner_attention.kv_caching(False)

        return final_rec
