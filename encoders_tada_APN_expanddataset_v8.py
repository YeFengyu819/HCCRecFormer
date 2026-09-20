"""
Encoder8：当前论文 full model 的精简实现。

这个文件只实现下面一个网络：
    t1_only_ownmask_tumorperi_v8_dense_roi_prompt_
    t2_region_pool_t2_cls_concat

阅读顺序就是前向计算顺序：
1. 加载 BiomedCLIP，并冻结原始 ViT blocks（terminal LayerNorm按Full模型保留可学习）；
2. 用 APN 将不规则 T1 帧对齐到规则时间轴；
3. 用同一对齐权重重采样 ROI mask，并生成 dense ROI prompt；
4. 用 TAdaFormer 和带时间戳的 BiLSTM 提取全局 T1 特征；
5. 将 T2 patch token 作为额外伪时间点加入肿瘤/瘤周 query pooling；
6. 拼接全局 T1、肿瘤、瘤周和 T2 CLS，完成二分类。

为了消除历史文件中的冗余，本文件不保留任何旧模型、消融分支或未启用模块。
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from Studies.HccEarlyRec_202602_paper.build_predict_models.videoseq_net.models.tada import (
    TAdaFormerBlock,
)


# =============================================================================
# 一、固定的网络规格
# =============================================================================

MODEL_NAME = (
    "t1_only_ownmask_tumorperi_v8_dense_roi_prompt_"
    "t2_region_pool_t2_cls_concat"
)
DEFAULT_BIOMEDCLIP_CHECKPOINT = (
    "/home/proteomics/DEEP_LEARNING_PRETRAINED_WEIGHTS/BIOMEDICAL/"
    "open_clip_pytorch_model.bin"
)
FEATURE_DIM = 768
TIME_EMBED_DIM = 16


# =============================================================================
# 二、时间输入与训练增强
# =============================================================================

def _to_2d_time(value, name):
    """把 (B,T,1) 或 (B,T) 时间张量统一成 (B,T)。"""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.dim() == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.dim() != 2:
        raise ValueError(f"{name} 必须是 (B,T) 或 (B,T,1)，实际为 {value.shape}")
    return value


def rebase_time_by_mask_anchor(time, time_mask, anchor_valid_index=1):
    """把指定顺位的有效帧设为 0 秒；有效帧不足时使用最后一个有效帧。"""
    if time is None or time_mask is None:
        return time, time_mask
    if anchor_valid_index < 0:
        raise ValueError("anchor_valid_index 不能小于 0")

    keep_last_dim = torch.is_tensor(time) and time.dim() == 3
    time_2d = _to_2d_time(time, "time")
    mask_2d = _to_2d_time(time_mask, "time_mask")
    if time_2d.shape != mask_2d.shape:
        raise ValueError("time 与 time_mask 的 B、T 维必须一致")

    dtype = time_2d.dtype if torch.is_floating_point(time_2d) else torch.float32
    time_2d = time_2d.to(dtype=dtype)
    valid = mask_2d.to(device=time_2d.device) > 0
    valid_count = valid.sum(dim=1)
    requested_rank = torch.full_like(valid_count, int(anchor_valid_index))
    anchor_rank = torch.minimum(requested_rank, (valid_count - 1).clamp_min(0))
    valid_rank = valid.long().cumsum(dim=1) - 1
    has_valid = valid_count > 0
    selected = valid & (valid_rank == anchor_rank[:, None]) & has_valid[:, None]
    anchor_time = (time_2d * selected.to(time_2d.dtype)).sum(dim=1)
    rebased = torch.where(
        has_valid[:, None],
        time_2d - anchor_time[:, None],
        time_2d,
    )
    return (rebased.unsqueeze(-1) if keep_last_dim else rebased), time_mask


def add_random_interframe_time_delay(
    time,
    time_mask,
    min_delay_seconds=0.0,
    max_delay_seconds=45.0,
):
    """随机延长第一帧之后的全部时间戳，用于模拟采集间隔变化。"""
    keep_last_dim = torch.is_tensor(time) and time.dim() == 3
    time_2d = _to_2d_time(time, "time")
    if min_delay_seconds < 0 or max_delay_seconds < min_delay_seconds:
        raise ValueError("帧间延迟范围不合法")
    if time_2d.shape[1] <= 1:
        return time, time_mask

    dtype = time_2d.dtype if torch.is_floating_point(time_2d) else torch.float32
    delayed = time_2d.to(dtype=dtype).clone()
    delay = delayed.new_empty(delayed.shape[0], 1).uniform_(
        float(min_delay_seconds),
        float(max_delay_seconds),
    )
    delayed[:, 1:] += delay
    return (delayed.unsqueeze(-1) if keep_last_dim else delayed), time_mask


def add_random_global_time_shift(
    time,
    time_mask,
    min_shift_seconds=-20.0,
    max_shift_seconds=20.0,
):
    """为每位患者的全部时间戳加入同一个随机平移量。"""
    keep_last_dim = torch.is_tensor(time) and time.dim() == 3
    time_2d = _to_2d_time(time, "time")
    if max_shift_seconds < min_shift_seconds:
        raise ValueError("全局时间平移范围不合法")

    dtype = time_2d.dtype if torch.is_floating_point(time_2d) else torch.float32
    shifted = time_2d.to(dtype=dtype)
    offset = shifted.new_empty(shifted.shape[0], 1).uniform_(
        float(min_shift_seconds),
        float(max_shift_seconds),
    )
    shifted = shifted + offset
    return (shifted.unsqueeze(-1) if keep_last_dim else shifted), time_mask


# =============================================================================
# 三、BiomedCLIP ViT 与 TAdaFormer
# =============================================================================

def _load_biomedclip_backbone(checkpoint_path):
    """创建 ViT-B/16，并严格加载 BiomedCLIP visual trunk 权重。"""
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    visual_state = {
        key.removeprefix("visual.trunk."): value
        for key, value in checkpoint.items()
        if key.startswith("visual.trunk.")
    }
    message = backbone.load_state_dict(visual_state, strict=True)
    print("BiomedCLIP loaded:", message)

    # 当前 full model 只训练 TAda、LoRA 和最终 LayerNorm，不微调原始 ViT block。
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in backbone.norm.parameters():
        parameter.requires_grad = True
    return backbone


class LoRALinear(nn.Module):
    """在线性层冻结权重旁增加低秩更新：Linear(x) + alpha/r * xAB。"""

    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def _copy_vit_block_to_tada(vit_block, tada_block):
    """把一个预训练 ViT block 的注意力、MLP 和 LayerNorm复制到 TAda block。"""
    tada_block.attn.in_proj_weight.data.copy_(vit_block.attn.qkv.weight.data)
    tada_block.attn.in_proj_bias.data.copy_(vit_block.attn.qkv.bias.data)
    tada_block.attn.out_proj.weight.data.copy_(vit_block.attn.proj.weight.data)
    tada_block.attn.out_proj.bias.data.copy_(vit_block.attn.proj.bias.data)
    tada_block.mlp[0].weight.data.copy_(vit_block.mlp.fc1.weight.data)
    tada_block.mlp[0].bias.data.copy_(vit_block.mlp.fc1.bias.data)
    tada_block.mlp[2].weight.data.copy_(vit_block.mlp.fc2.weight.data)
    tada_block.mlp[2].bias.data.copy_(vit_block.mlp.fc2.bias.data)
    tada_block.ln_1.weight.data.copy_(vit_block.norm1.weight.data)
    tada_block.ln_1.bias.data.copy_(vit_block.norm1.bias.data)
    tada_block.ln_2.weight.data.copy_(vit_block.norm2.weight.data)
    tada_block.ln_2.bias.data.copy_(vit_block.norm2.bias.data)


class TAdaEncoder(nn.Module):
    """将指定的 ViT block 替换为带双 TAda adapter 的时序 block。"""

    def __init__(
        self,
        backbone,
        num_frames,
        start_layer,
        end_layer,
        use_lora=False,
    ):
        super().__init__()
        if end_layer <= start_layer:
            raise ValueError("TAdaEncoder 至少需要一个 block")

        self.blocks = nn.ModuleList(
            TAdaFormerBlock(
                d_model=FEATURE_DIM,
                n_head=12,
                num_frames=num_frames,
                drop_path=0.1,
                attn_dropout=0,
                reduction=2,
                rf_r=2,
                rf_k=[3, 3],
                temporal_enhance=True,
                double_tada=True,
            )
            for _ in range(end_layer - start_layer)
        )
        for index, block in enumerate(self.blocks):
            _copy_vit_block_to_tada(backbone.blocks[start_layer + index], block)
            if use_lora:
                block.mlp[0] = LoRALinear(block.mlp[0], rank=8, alpha=16)
                block.mlp[2] = LoRALinear(block.mlp[2], rank=8, alpha=16)

        # 冻结复制进来的 ViT 主体，只训练两个 TAda adapter 和可选 LoRA 参数。
        for parameter in self.parameters():
            parameter.requires_grad = False
        for block in self.blocks:
            for parameter in block.tada.parameters():
                parameter.requires_grad = True
            for parameter in block.tada2.parameters():
                parameter.requires_grad = True
        if use_lora:
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        mode = "TAda + LoRA" if use_lora else "TAda only"
        print(f"{mode}: trainable {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    def forward(self, tokens):
        # TAdaFormerBlock 使用 (token, batch*frame, channel) 排布。
        tokens = tokens.permute(1, 0, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.permute(1, 0, 2)


# =============================================================================
# 四、APN：将不规则观测帧对齐到规则时间轴
# =============================================================================

class TemporalAlignment(nn.Module):
    """从真实时间戳构造平滑对齐权重，并同步聚合图像token和Time2Vec。"""

    _MODE_ALIASES = {
        "fixed": "fixed",
        "window": "window",
        "tau": "tau",
        "window_tau": "window_tau",
    }

    def __init__(
        self,
        start_time,
        end_time,
        num_regular_frames,
        fuse_window,
        kernel_sigma,
        learn_mode,
        fuse_window_offset_min,
        fuse_window_offset_max,
        fuse_window_offset_init,
        tau_multiplier_init,
    ):
        super().__init__()
        self.fuse_window = float(fuse_window)
        self.kernel_sigma = float(kernel_sigma)
        self.eps = 1e-6
        self.learn_mode = self._MODE_ALIASES.get(str(learn_mode).lower())
        if self.learn_mode is None:
            raise ValueError("apn_learn_mode 仅支持 fixed/window/tau/window_tau")
        if self.fuse_window <= 0 or self.kernel_sigma <= 0:
            raise ValueError("fuse_window 与 kernel_sigma 必须大于 0")

        self.raw_fuse_window_offset = None
        if self.learn_mode in {"window", "window_tau"}:
            lower = -0.5 * self.fuse_window if fuse_window_offset_min is None else float(fuse_window_offset_min)
            upper = 0.5 * self.fuse_window if fuse_window_offset_max is None else float(fuse_window_offset_max)
            initial = float(fuse_window_offset_init)
            if not lower <= initial <= upper or upper <= lower:
                raise ValueError("APN窗口偏移边界或初值不合法")
            self.fuse_window_offset_min = lower
            self.fuse_window_offset_max = upper
            ratio = min(max((initial - lower) / (upper - lower), 1e-6), 1 - 1e-6)
            self.raw_fuse_window_offset = nn.Parameter(
                torch.tensor(math.log(ratio / (1 - ratio)))
            )
        else:
            self.fuse_window_offset_min = None
            self.fuse_window_offset_max = None

        self.raw_tau_multiplier = None
        if self.learn_mode in {"tau", "window_tau"}:
            self.raw_tau_multiplier = nn.Parameter(
                torch.tensor(math.log(max(float(tau_multiplier_init), self.eps)))
            )

        # Time2Vec：第一维为线性分量，其余15维为周期分量。
        self.time_scale = nn.Linear(1, 1)
        self.time_periodic = nn.Linear(1, TIME_EMBED_DIM - 1)
        self.register_buffer(
            "regular_times",
            torch.linspace(start_time, end_time, steps=num_regular_frames),
        )

    def _current_window(self, reference):
        if self.raw_fuse_window_offset is None:
            return reference.new_tensor(self.fuse_window)
        lower = self.fuse_window_offset_min
        upper = self.fuse_window_offset_max
        offset = lower + (upper - lower) * torch.sigmoid(self.raw_fuse_window_offset)
        return (reference.new_tensor(self.fuse_window) + offset).clamp_min(self.eps)

    def _current_sigma(self, reference):
        if self.raw_tau_multiplier is None:
            return reference.new_tensor(self.kernel_sigma)
        return (
            reference.new_tensor(self.kernel_sigma)
            * torch.exp(self.raw_tau_multiplier)
        ).clamp_min(self.eps)

    def compute_alpha(self, real_times, regular_times=None):
        """计算每个规则锚点对每个真实帧的平滑窗口权重。"""
        if regular_times is None:
            regular_times = self.regular_times
        regular = regular_times.to(real_times).view(1, -1, 1)
        observed = real_times.unsqueeze(1)
        window = self._current_window(real_times)
        sigma = self._current_sigma(real_times)
        left = regular - window
        right = regular + window
        return torch.sigmoid((right - observed) / sigma) * torch.sigmoid(
            (observed - left) / sigma
        )

    def forward(self, tokens, time, time_mask):
        time = _to_2d_time(time, "time").to(device=tokens.device)
        if not torch.is_floating_point(time):
            time = time.float()
        mask = _to_2d_time(time_mask, "time_mask").to(
            device=tokens.device,
            dtype=time.dtype,
        )
        batch, observed_frames = time.shape
        tokens = tokens.view(batch, observed_frames, tokens.shape[1], tokens.shape[2])

        alpha = self.compute_alpha(time)
        alpha = alpha * mask.unsqueeze(1)
        alpha_norm = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
        regular_tokens = torch.einsum("brt,btnc->brnc", alpha_norm, tokens)

        time_input = time.unsqueeze(-1)
        time_embedding = torch.cat(
            [self.time_scale(time_input), torch.sin(self.time_periodic(time_input))],
            dim=-1,
        )
        regular_time = torch.einsum("brt,btd->brd", alpha_norm, time_embedding)
        return regular_tokens, regular_time, alpha_norm


# =============================================================================
# 五、ROI mask：统一形状、训练期轮廓扰动和dense prompt
# =============================================================================

def normalize_frame_mask(mask, batch_size, num_frames, name):
    """把常见mask排布统一成 (B,T,1,H,W)，并限制在[0,1]。"""
    if mask is None:
        raise ValueError(f"{name} 不能为空")
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.dim() == 3:
        mask = mask[:, None, None]
    elif mask.dim() == 4:
        if mask.shape[1] == num_frames:
            mask = mask[:, :, None]
        elif mask.shape[1] == 1:
            mask = mask[:, None]
        else:
            raise ValueError(
                f"{name} 的4维输入必须是(B,T,H,W)或(B,1,H,W)，实际为{mask.shape}"
            )
    elif mask.dim() == 5 and mask.shape[2] == num_frames:
        mask = mask.permute(0, 2, 1, 3, 4)
    elif mask.dim() != 5:
        raise ValueError(f"{name} 的维度不合法：{mask.shape}")

    if mask.shape[0] != batch_size:
        raise ValueError(f"{name} 的batch与图像不一致")
    if mask.shape[1] == 1 and num_frames > 1:
        mask = mask.expand(batch_size, num_frames, *mask.shape[2:])
    if mask.shape[1] != num_frames:
        raise ValueError(f"{name} 的帧数不是 {num_frames}：{mask.shape}")
    mask = mask.float()
    if mask.shape[2] != 1:
        mask = mask.max(dim=2, keepdim=True).values
    return mask.clamp(0.0, 1.0)


def jitter_roi_mask(mask, probability, radius_ratio):
    """以帧为单位随机膨胀或腐蚀轮廓；只在训练阶段由主模型调用。"""
    batch_frames, _, height, width = mask.shape
    non_empty = mask.flatten(1).any(dim=1)
    apply = (torch.rand(batch_frames, device=mask.device) < probability) & non_empty
    if not bool(apply.any()):
        return mask

    max_radius = max(1, int(round(min(height, width) * radius_ratio)))
    radius = torch.randint(1, max_radius + 1, (batch_frames,), device=mask.device)
    dilate = torch.rand(batch_frames, device=mask.device) < 0.5
    output = mask.clone()
    for current_radius in range(1, max_radius + 1):
        kernel = 2 * current_radius + 1
        dilate_index = apply & dilate & (radius == current_radius)
        if bool(dilate_index.any()):
            output[dilate_index] = F.max_pool2d(
                mask[dilate_index], kernel, stride=1, padding=current_radius
            )

        erode_index = apply & (~dilate) & (radius == current_radius)
        if bool(erode_index.any()):
            eroded = 1 - F.max_pool2d(
                1 - mask[erode_index], kernel, stride=1, padding=current_radius
            )
            empty = ~eroded.flatten(1).any(dim=1)
            eroded[empty] = mask[erode_index][empty]
            output[erode_index] = eroded
    return output


class DenseROIPromptEncoder(nn.Module):
    """把APN对齐后的肿瘤/瘤周概率图编码为patch级空间prompt。"""

    def __init__(self, hidden_dim, dropout, init_scale, mask_threshold):
        super().__init__()
        groups = next((value for value in (8, 4, 2) if hidden_dim % value == 0), 1)
        self.mask_threshold = float(mask_threshold)
        self.prompt_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.prompt_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, FEATURE_DIM, 1, bias=False),
        )
        self.out_norm = nn.LayerNorm(FEATURE_DIM)

    def forward(self, tumor_mask, peri_mask, alpha_norm, grid_size):
        batch, regular_frames, observed_frames = alpha_norm.shape
        tumor = normalize_frame_mask(tumor_mask, batch, observed_frames, "T1肿瘤mask")
        peri = normalize_frame_mask(peri_mask, batch, observed_frames, "T1瘤周mask")
        tumor = tumor.to(alpha_norm.device)
        peri = peri.to(alpha_norm.device)
        height, width = tumor.shape[-2:]

        observed = torch.cat([tumor, peri], dim=2).reshape(
            batch * observed_frames, 2, height, width
        )
        observed_probs = F.interpolate(
            observed,
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        observed_probs = observed_probs.view(
            batch, observed_frames, 2, grid_size, grid_size
        )
        regular_probs = torch.einsum(
            "brt,btkhw->brkhw",
            alpha_norm.to(observed_probs),
            observed_probs,
        ).clamp(0.0, 1.0)

        # V8定义允许肿瘤与瘤周区域重叠，两路mask分别阈值化。
        regular_masks = regular_probs >= self.mask_threshold
        prompt_input = regular_probs * regular_masks.to(regular_probs.dtype)
        prompt_map = self.prompt_net(
            prompt_input.reshape(batch * regular_frames, 2, grid_size, grid_size)
        )
        support = prompt_input.reshape(
            batch * regular_frames, 2, grid_size, grid_size
        ).amax(dim=1, keepdim=True)
        prompt_tokens = self.out_norm(prompt_map.flatten(2).transpose(1, 2))
        prompt_tokens = prompt_tokens * support.flatten(2).transpose(1, 2)
        prompt_tokens = prompt_tokens.view(
            batch, regular_frames, grid_size * grid_size, FEATURE_DIM
        )
        return prompt_tokens, observed_probs, regular_probs, regular_masks


# =============================================================================
# 六、T1主分支：ViT前8层、APN、ROI prompt、TAda、时间戳BiLSTM
# =============================================================================

class T1TemporalEncoder(nn.Module):
    """提取全局T1描述符和规则时间轴上的patch tokens。"""

    def __init__(
        self,
        checkpoint_path,
        num_input_frames,
        num_regular_frames,
        start_time,
        end_time,
        tada_start,
        tada_end,
        fuse_window,
        kernel_sigma,
        apn_learn_mode,
        fuse_window_offset_min,
        fuse_window_offset_max,
        fuse_window_offset_init,
        tau_multiplier_init,
        cls_lstm_hidden_dim,
        cls_lstm_num_layers,
        cls_lstm_bidirectional,
        cls_lstm_dropout,
        roi_prompt_hidden_dim,
        roi_prompt_dropout,
        roi_prompt_init_scale,
        roi_mask_token_threshold,
        background_dropout_enabled,
        background_dropout_ratio,
        background_dropout_apply_prob,
    ):
        super().__init__()
        if tada_end != 12 or not 0 < tada_start < 11:
            raise ValueError("当前full模型要求 tada_start<11 且 tada_end=12")
        self.num_input_frames = int(num_input_frames)
        self.num_regular_frames = int(num_regular_frames)
        self.spatial_end_depth = int(tada_start)
        self.feat_dim = FEATURE_DIM
        self.backbone = _load_biomedclip_backbone(checkpoint_path)
        self.apn = TemporalAlignment(
            start_time=start_time,
            end_time=end_time,
            num_regular_frames=num_regular_frames,
            fuse_window=fuse_window,
            kernel_sigma=kernel_sigma,
            learn_mode=apn_learn_mode,
            fuse_window_offset_min=fuse_window_offset_min,
            fuse_window_offset_max=fuse_window_offset_max,
            fuse_window_offset_init=fuse_window_offset_init,
            tau_multiplier_init=tau_multiplier_init,
        )
        self.temporal_encoder1 = TAdaEncoder(
            self.backbone,
            num_frames=num_regular_frames,
            start_layer=tada_start,
            end_layer=11,
            use_lora=False,
        )
        self.temporal_encoder2 = TAdaEncoder(
            self.backbone,
            num_frames=num_regular_frames,
            start_layer=11,
            end_layer=tada_end,
            use_lora=True,
        )

        # 当前full模型把16维规则时间嵌入投影到768维，并加到全部规则token。
        self.token_time_proj = nn.Sequential(
            nn.Linear(TIME_EMBED_DIM, FEATURE_DIM),
            nn.LayerNorm(FEATURE_DIM),
        )
        self.token_time_scale = nn.Parameter(torch.tensor(0.1))

        # 同一16维时间嵌入还会与最终CLS拼接，进入双向LSTM。
        lstm_dropout = cls_lstm_dropout if cls_lstm_num_layers > 1 else 0.0
        self.cls_lstm = nn.LSTM(
            input_size=FEATURE_DIM + TIME_EMBED_DIM,
            hidden_size=cls_lstm_hidden_dim,
            num_layers=cls_lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=cls_lstm_bidirectional,
        )
        lstm_output_dim = cls_lstm_hidden_dim * (2 if cls_lstm_bidirectional else 1)
        self.cls_lstm_proj = (
            nn.Identity()
            if lstm_output_dim == FEATURE_DIM
            else nn.Linear(lstm_output_dim, FEATURE_DIM)
        )
        self.roi_prompt_encoder = DenseROIPromptEncoder(
            hidden_dim=roi_prompt_hidden_dim,
            dropout=roi_prompt_dropout,
            init_scale=roi_prompt_init_scale,
            mask_threshold=roi_mask_token_threshold,
        )

        self.background_dropout_enabled = bool(background_dropout_enabled)
        self.background_dropout_ratio = float(background_dropout_ratio)
        self.background_dropout_apply_prob = float(background_dropout_apply_prob)
        if not 0 <= self.background_dropout_ratio < 1:
            raise ValueError("background_dropout_ratio 必须位于 [0,1)")
        if not 0 <= self.background_dropout_apply_prob <= 1:
            raise ValueError("background_dropout_apply_prob 必须位于 [0,1]")

    def _arrange_t1_frames(self, images):
        """接受(B,T,3,H,W)或(B,3,T,H,W)，输出(B*T,3,H,W)。"""
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"T1输入必须是4或5维，实际为 {images.shape}")
        if images.shape[1] == self.num_input_frames:
            images = images.permute(0, 2, 1, 3, 4)
        elif images.shape[2] != self.num_input_frames:
            raise ValueError(f"T1输入中找不到 {self.num_input_frames} 帧")

        batch, channels, frames, height, width = images.shape
        if channels != 3 or frames != self.num_input_frames:
            raise ValueError(f"T1输入应为3通道、{self.num_input_frames}帧")
        flat = images.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        return flat, batch, frames

    def _embed_images(self, images):
        """完成patch embedding、CLS和位置编码。"""
        tokens = self.backbone.patch_embed(images)
        cls = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.backbone.pos_drop(tokens + self.backbone.pos_embed)

    def encode_t2(self, t2):
        """让T2通过完整共享ViT，返回归一化后的[CLS+patch] tokens。"""
        if t2.dim() == 3:
            t2 = t2.unsqueeze(1)
        if t2.dim() != 4:
            raise ValueError(f"T2输入必须是(B,C,H,W)，实际为 {t2.shape}")
        if t2.shape[1] == 1:
            t2 = t2.repeat(1, 3, 1, 1)
        if t2.shape[1] != 3:
            raise ValueError("T2必须是1或3通道")

        tokens = self._embed_images(t2)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        return self.backbone.norm(tokens)

    def _add_dense_prompt(self, tokens, prompt):
        """只给patch token增加prompt，CLS保持不变。"""
        scale = self.roi_prompt_encoder.prompt_scale.to(tokens)
        cls = tokens[:, :, :1]
        patches = tokens[:, :, 1:] + scale * prompt.to(tokens)
        return torch.cat([cls, patches], dim=2)

    def _drop_background_tokens(self, tokens, tumor_mask):
        """训练时随机清零非肿瘤背景patch；CLS与肿瘤patch始终保留。"""
        if not (
            self.training
            and self.background_dropout_enabled
            and self.background_dropout_ratio > 0
            and self.background_dropout_apply_prob > 0
        ):
            return tokens
        batch, regular_frames, token_count, _ = tokens.shape
        apply = torch.rand(batch, regular_frames, 1, device=tokens.device)
        apply = apply < self.background_dropout_apply_prob
        drop = torch.rand(
            batch,
            regular_frames,
            token_count - 1,
            device=tokens.device,
        ) < self.background_dropout_ratio
        drop = drop & (~tumor_mask.to(tokens.device)) & apply
        patches = tokens[:, :, 1:] * (~drop).to(tokens.dtype).unsqueeze(-1)
        return torch.cat([tokens[:, :, :1], patches], dim=2)

    def forward(self, t1, tumor_mask, peri_mask, time, time_mask):
        """返回全局T1描述符、规则patch tokens以及对齐后的ROI信息。"""
        frames, batch, observed_frames = self._arrange_t1_frames(t1)
        time = _to_2d_time(time, "time").to(frames.device)
        time_mask = _to_2d_time(time_mask, "time_mask").to(frames.device)
        if time.shape != (batch, observed_frames) or time_mask.shape != time.shape:
            raise ValueError("T1图像、time和time_mask的B、T维必须一致")

        tokens = self._embed_images(frames)
        for block in self.backbone.blocks[: self.spatial_end_depth]:
            tokens = block(tokens)
        tokens, regular_time, alpha_norm = self.apn(tokens, time, time_mask)

        patch_count = tokens.shape[2] - 1
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("ViT patch token数量不是平方数")
        prompt, observed_probs, regular_probs, regular_masks = self.roi_prompt_encoder(
            tumor_mask,
            peri_mask,
            alpha_norm,
            grid_size,
        )
        tokens = self._add_dense_prompt(tokens, prompt)

        # 时间嵌入先加入token，再做背景dropout，避免被清零的背景重新获得时间信号。
        token_time = self.token_time_proj(regular_time)[:, :, None]
        tokens = tokens + self.token_time_scale * token_time
        tumor_token_mask = regular_masks[:, :, 0].flatten(2)
        peri_token_mask = regular_masks[:, :, 1].flatten(2)
        tokens = self._drop_background_tokens(tokens, tumor_token_mask)

        batch, regular_frames, token_count, channels = tokens.shape
        tokens = tokens.reshape(batch * regular_frames, token_count, channels)
        tokens = self.temporal_encoder1(tokens)
        tokens = self.temporal_encoder2(tokens)
        self.debug_tada_output = tokens
        tokens = self.backbone.norm(tokens)

        cls_sequence = tokens[:, 0].reshape(batch, regular_frames, FEATURE_DIM)
        patch_tokens = tokens[:, 1:].reshape(
            batch, regular_frames, token_count - 1, FEATURE_DIM
        )
        lstm_input = torch.cat([cls_sequence, regular_time], dim=-1)
        lstm_sequence, _ = self.cls_lstm(lstm_input)
        global_t1 = self.cls_lstm_proj(lstm_sequence.mean(dim=1))

        aux = {
            "roi_prompt_fusion_location": "post_apn",
            "roi_prompt_scale": self.roi_prompt_encoder.prompt_scale.detach(),
            "observed_prompt_probs": observed_probs,
            "regular_prompt_probs": regular_probs,
            "regular_tumor_token_mask": tumor_token_mask,
            "regular_peri_token_mask": peri_token_mask,
            "alpha_norm": alpha_norm,
            "regular_time_emb": regular_time,
        }
        return global_t1, patch_tokens, aux


# =============================================================================
# 七、肿瘤/瘤周query pooling
# =============================================================================

class FrameMaskedRegionSTPool(nn.Module):
    """用可学习query在指定ROI支持内聚合全部T1规则帧和T2伪时间点。"""

    def __init__(self, num_queries, num_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, FEATURE_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=FEATURE_DIM,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.proj = nn.Sequential(
            nn.Linear(FEATURE_DIM * num_queries, FEATURE_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
        )

    def forward(self, tokens, region_mask):
        batch, frames, patches, channels = tokens.shape
        if region_mask.shape != (batch, frames, patches):
            raise ValueError("区域mask与patch token形状不一致")
        flat_tokens = tokens.reshape(batch, frames * patches, channels)
        flat_mask = region_mask.bool().reshape(batch, frames * patches)
        valid = flat_mask.any(dim=1)
        padding_mask = ~flat_mask
        padding_mask[~valid] = False

        queries = self.query.expand(batch, -1, -1)
        query_features, attention = self.attn(
            query=queries,
            key=flat_tokens,
            value=flat_tokens,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        valid_float = valid.to(query_features.dtype)[:, None, None]
        query_features = self.norm(query_features * valid_float)
        attention = attention * valid_float
        region_feature = self.proj(query_features.reshape(batch, -1))
        return region_feature, attention


# =============================================================================
# 八、完整模型：T1全局 + T1/T2区域 + T2全局
# =============================================================================

class T1_only_OwnMaskTumorPeri_v8_DenseROIPrompt_T2RegionPool(nn.Module):
    """论文full模型；类名保留，便于现有训练与分析脚本迁移。"""

    def __init__(
        self,
        checkpoint_path=DEFAULT_BIOMEDCLIP_CHECKPOINT,
        num_input_frames=8,
        num_regular_frames=8,
        start_time=0,
        end_time=128,
        tada_start=8,
        tada_end=12,
        fuse_window=64,
        kernel_sigma=5.0,
        apn_learn_mode="window",
        fuse_window_offset_min=0,
        fuse_window_offset_max=16,
        fuse_window_offset_init=0,
        tau_multiplier_init=1.0,
        cls_lstm_hidden_dim=64,
        cls_lstm_num_layers=1,
        cls_lstm_bidirectional=True,
        cls_lstm_dropout=0.0,
        roi_prompt_hidden_dim=64,
        roi_prompt_dropout=0.1,
        roi_prompt_init_scale=1e-3,
        roi_mask_token_threshold=0.1,
        roi_mask_jitter_enabled=True,
        roi_mask_jitter_prob=0.5,
        roi_mask_jitter_ratio=0.01,
        background_dropout_enabled=True,
        background_dropout_ratio=0.8,
        background_dropout_apply_prob=0.5,
        t2_pool_mask_threshold=0.1,
        t2_pool_apply_mask_jitter=True,
        num_heads=4,
        tumor_num_queries=4,
        peri_num_queries=2,
        dropout=0.2,
        rebase_time=False,
        time_anchor_valid_index=1,
        add_random_interframe_time_delay_enabled=True,
        add_random_global_time_shift_enabled=True,
        global_time_shift_min_seconds=-45.0,
        global_time_shift_max_seconds=45.0,
    ):
        super().__init__()
        self.encoder = T1TemporalEncoder(
            checkpoint_path=checkpoint_path,
            num_input_frames=num_input_frames,
            num_regular_frames=num_regular_frames,
            start_time=start_time,
            end_time=end_time,
            tada_start=tada_start,
            tada_end=tada_end,
            fuse_window=fuse_window,
            kernel_sigma=kernel_sigma,
            apn_learn_mode=apn_learn_mode,
            fuse_window_offset_min=fuse_window_offset_min,
            fuse_window_offset_max=fuse_window_offset_max,
            fuse_window_offset_init=fuse_window_offset_init,
            tau_multiplier_init=tau_multiplier_init,
            cls_lstm_hidden_dim=cls_lstm_hidden_dim,
            cls_lstm_num_layers=cls_lstm_num_layers,
            cls_lstm_bidirectional=cls_lstm_bidirectional,
            cls_lstm_dropout=cls_lstm_dropout,
            roi_prompt_hidden_dim=roi_prompt_hidden_dim,
            roi_prompt_dropout=roi_prompt_dropout,
            roi_prompt_init_scale=roi_prompt_init_scale,
            roi_mask_token_threshold=roi_mask_token_threshold,
            background_dropout_enabled=background_dropout_enabled,
            background_dropout_ratio=background_dropout_ratio,
            background_dropout_apply_prob=background_dropout_apply_prob,
        )
        self.tumor_pool = FrameMaskedRegionSTPool(
            num_queries=tumor_num_queries,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.peri_pool = FrameMaskedRegionSTPool(
            num_queries=peri_num_queries,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cls_with_t2 = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM * 4),
            nn.Linear(FEATURE_DIM * 4, FEATURE_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FEATURE_DIM, 1),
        )
        self.feat_dim = FEATURE_DIM * 4

        self.roi_mask_jitter_enabled = bool(roi_mask_jitter_enabled)
        self.roi_mask_jitter_prob = float(roi_mask_jitter_prob)
        self.roi_mask_jitter_ratio = float(roi_mask_jitter_ratio)
        self.t2_pool_mask_threshold = float(t2_pool_mask_threshold)
        self.t2_pool_apply_mask_jitter = bool(t2_pool_apply_mask_jitter)
        self.rebase_time = bool(rebase_time)
        self.time_anchor_valid_index = int(time_anchor_valid_index)
        self.add_random_interframe_time_delay_enabled = bool(
            add_random_interframe_time_delay_enabled
        )
        self.add_random_global_time_shift_enabled = bool(
            add_random_global_time_shift_enabled
        )
        self.global_time_shift_min_seconds = float(global_time_shift_min_seconds)
        self.global_time_shift_max_seconds = float(global_time_shift_max_seconds)

    def _augment_time(self, time, time_mask, device):
        """按full配置依次执行time rebase、帧间延迟和全局时间平移。"""
        time = _to_2d_time(time, "time").to(device)
        time_mask = _to_2d_time(time_mask, "time_mask").to(device)
        if self.rebase_time:
            time, time_mask = rebase_time_by_mask_anchor(
                time,
                time_mask,
                self.time_anchor_valid_index,
            )
        if (
            self.training
            and self.add_random_interframe_time_delay_enabled
            and torch.rand((), device=device) < 0.5
        ):
            time, time_mask = add_random_interframe_time_delay(
                time,
                time_mask,
                min_delay_seconds=0.0,
                max_delay_seconds=45.0,
            )
        if self.training and self.add_random_global_time_shift_enabled:
            time, time_mask = add_random_global_time_shift(
                time,
                time_mask,
                self.global_time_shift_min_seconds,
                self.global_time_shift_max_seconds,
            )
        return time, time_mask

    def _augment_mask(self, mask, batch, frames, name, device):
        """统一mask形状，并在训练时模拟轮廓不确定性。"""
        normalized = normalize_frame_mask(mask, batch, frames, name).to(device)
        if not (
            self.training
            and self.roi_mask_jitter_enabled
            and self.roi_mask_jitter_prob > 0
            and self.roi_mask_jitter_ratio > 0
        ):
            return normalized
        flat = (normalized.reshape(batch * frames, 1, *normalized.shape[-2:]) >= 0.5).float()
        flat = jitter_roi_mask(
            flat,
            probability=self.roi_mask_jitter_prob,
            radius_ratio=self.roi_mask_jitter_ratio,
        )
        return flat.view_as(normalized)

    def _prepare_t2_masks(self, tumor_mask, peri_mask, batch, grid_size, device):
        """将T2肿瘤/瘤周mask降采样到ViT patch网格并分别阈值化。"""
        if tumor_mask is None and peri_mask is None:
            raise ValueError("T2肿瘤mask和瘤周mask不能同时为空")
        if tumor_mask is None:
            tumor_mask = torch.zeros_like(peri_mask)
        if peri_mask is None:
            peri_mask = torch.zeros_like(tumor_mask)

        if self.training and self.t2_pool_apply_mask_jitter and self.roi_mask_jitter_enabled:
            tumor = self._augment_mask(tumor_mask, batch, 1, "T2肿瘤mask", device)
            peri = self._augment_mask(peri_mask, batch, 1, "T2瘤周mask", device)
        else:
            tumor = normalize_frame_mask(tumor_mask, batch, 1, "T2肿瘤mask").to(device)
            peri = normalize_frame_mask(peri_mask, batch, 1, "T2瘤周mask").to(device)

        def to_token_mask(mask):
            height, width = mask.shape[-2:]
            probability = F.interpolate(
                mask.reshape(batch, 1, height, width),
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=False,
            ).flatten(1).view(batch, 1, grid_size * grid_size).clamp(0.0, 1.0)
            return probability >= self.t2_pool_mask_threshold, probability

        tumor_binary, tumor_probability = to_token_mask(tumor)
        peri_binary, peri_probability = to_token_mask(peri)
        return tumor_binary, peri_binary, tumor_probability, peri_probability

    def get_optimizer_param_groups(self, base_lr, classifier_lr):
        """分类头使用独立学习率，其余可训练参数使用基础学习率。"""
        classifier_parameters = [
            parameter
            for parameter in self.cls_with_t2.parameters()
            if parameter.requires_grad
        ]
        classifier_ids = {id(parameter) for parameter in classifier_parameters}
        other_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in classifier_ids
        ]
        return [
            {
                "params": other_parameters,
                "lr": float(base_lr),
                "group_name": "non_classifier",
            },
            {
                "params": classifier_parameters,
                "lr": float(classifier_lr),
                "group_name": "v8_t2_region_pool_cls_classifier",
            },
        ]

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧encoder7 checkpoint，同时丢弃其中未参与full前向的冗余参数。"""
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = OrderedDict(
                (key.removeprefix("module."), value)
                for key, value in state_dict.items()
            )

        ignored_prefixes = (
            "encoder.spatial_encoder.backbone.",  # 旧文件重复注册的backbone副本
            "encoder.cls_time_proj.",             # time2vec='tokens'时从不调用
            "cls.",                               # 旧T1-only分类头在full模型中从不调用
        )
        ignored_keys = {"encoder.cls_lstm_res_scale"}  # residual=False时从不调用
        cleaned = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key not in ignored_keys
            and not any(key.startswith(prefix) for prefix in ignored_prefixes)
        )
        return super().load_state_dict(cleaned, strict=strict)

    def forward(
        self,
        ce,
        ce_mask_tumor,
        ce_mask_peri,
        t2,
        t2_mask_tumor,
        t2_mask_peri,
        time,
        time_mask,
        clinic,
        return_attn=False,
    ):
        """保持encoder7的调用接口；clinic在当前网络中不参与计算。"""
        del clinic
        if ce.dim() != 5:
            raise ValueError("当前full模型要求T1为5维多帧输入")
        batch = ce.shape[0]
        t1_frames = self.encoder.num_input_frames
        device = ce.device
        time, time_mask = self._augment_time(time, time_mask, device)
        t1_tumor = self._augment_mask(
            ce_mask_tumor, batch, t1_frames, "T1肿瘤mask", device
        )
        t1_peri = self._augment_mask(
            ce_mask_peri, batch, t1_frames, "T1瘤周mask", device
        )

        global_t1, t1_patch_tokens, aux = self.encoder(
            ce,
            tumor_mask=t1_tumor,
            peri_mask=t1_peri,
            time=time,
            time_mask=time_mask,
        )
        regular_frames, patch_count, channels = t1_patch_tokens.shape[1:]
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("T1 patch token数量不是平方数")

        t2_tokens = self.encoder.encode_t2(t2)
        t2_cls = t2_tokens[:, 0]
        t2_patches = t2_tokens[:, 1:]
        if t2_patches.shape != (batch, patch_count, channels):
            raise ValueError("T1与T2的patch token形状不一致")

        t2_tumor, t2_peri, t2_tumor_prob, t2_peri_prob = self._prepare_t2_masks(
            t2_mask_tumor,
            t2_mask_peri,
            batch,
            grid_size,
            device,
        )
        t1_tumor_mask = aux["regular_tumor_token_mask"]
        t1_peri_mask = aux["regular_peri_token_mask"]

        # T2 patch网格作为第R+1个伪时间点加入两路区域池化。
        pool_tokens = torch.cat([t1_patch_tokens, t2_patches[:, None]], dim=1)
        tumor_mask = torch.cat([t1_tumor_mask, t2_tumor], dim=1)
        peri_mask = torch.cat([t1_peri_mask, t2_peri], dim=1)
        tumor_feature, tumor_attention = self.tumor_pool(pool_tokens, tumor_mask)
        peri_feature, peri_attention = self.peri_pool(pool_tokens, peri_mask)

        feature = torch.cat(
            [global_t1, tumor_feature, peri_feature, t2_cls],
            dim=-1,
        )
        prediction = self.cls_with_t2(feature)
        if not return_attn:
            return prediction, feature

        t1_flat_count = regular_frames * patch_count
        aux.update(
            {
                "use_tumor_peritumor_pool": True,
                "roi_mask_jitter_enabled": self.roi_mask_jitter_enabled,
                "roi_masks_are_nonexclusive": True,
                "t2_region_pool_enabled": True,
                "concat_t2_cls": True,
                "t2_prompt_generation": False,
                "t2_pool_mask_threshold": self.t2_pool_mask_threshold,
                "t2_pool_apply_mask_jitter": self.t2_pool_apply_mask_jitter,
                "tumor_attn": tumor_attention,
                "peri_attn": peri_attention,
                "tumor_attn_t1": tumor_attention[..., :t1_flat_count].reshape(
                    batch, tumor_attention.shape[1], regular_frames, patch_count
                ),
                "tumor_attn_t2": tumor_attention[..., t1_flat_count:],
                "peri_attn_t1": peri_attention[..., :t1_flat_count].reshape(
                    batch, peri_attention.shape[1], regular_frames, patch_count
                ),
                "peri_attn_t2": peri_attention[..., t1_flat_count:],
                "tumor_token_mask": tumor_mask,
                "peri_token_mask": peri_mask,
                "peri_token_mask_before_dropout": peri_mask,
                "t1_tumor_token_mask": t1_tumor_mask,
                "t1_peri_token_mask": t1_peri_mask,
                "t2_tumor_token_mask": t2_tumor,
                "t2_peri_token_mask": t2_peri,
                "t2_tumor_token_probs": t2_tumor_prob,
                "t2_peri_token_probs": t2_peri_prob,
                "t2_patch_tokens": t2_patches,
                "t2_cls": t2_cls,
                "pool_num_t1_frames": regular_frames,
                "pool_t2_frame_index": regular_frames,
                "t2_fusion_location": "tumor_peritumor_pool_and_classifier",
            }
        )
        return prediction, feature, aux


# =============================================================================
# 九、唯一的模型工厂
# =============================================================================

def create_model(cfg):
    """从现有Config字典创建唯一支持的full模型。"""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") != MODEL_NAME:
        raise ValueError(f"encoder8只支持模型 {MODEL_NAME}")
    if model_cfg.get("roi_prompt_fusion_location", "post_apn") != "post_apn":
        raise ValueError("encoder8只保留full模型实际使用的post_apn ROI prompt")
    if model_cfg.get("cls_pooling", "lstm_with_timestamp") != "lstm_with_timestamp":
        raise ValueError("encoder8只保留full模型实际使用的lstm_with_timestamp")
    if model_cfg.get("mode", "fix_vit") != "fix_vit":
        raise ValueError("encoder8只保留full模型实际使用的fix_vit策略")
    if not model_cfg.get("use_tumor_peritumor_pool", True):
        raise ValueError("encoder8的full模型必须启用tumor/peritumor pooling")
    if not model_cfg.get("concat_t2_cls", True):
        raise ValueError("encoder8的full模型必须拼接T2 CLS")

    locations = model_cfg.get("background_dropout_locations", ["post_apn"])
    if isinstance(locations, str):
        locations = [locations]
    if set(locations) - {"post_apn"}:
        raise ValueError("encoder8只保留full配置实际使用的post_apn background dropout")

    return T1_only_OwnMaskTumorPeri_v8_DenseROIPrompt_T2RegionPool(
        checkpoint_path=model_cfg.get(
            "checkpoint_path",
            DEFAULT_BIOMEDCLIP_CHECKPOINT,
        ),
        num_input_frames=model_cfg.get("num_input_frames", 8),
        num_regular_frames=model_cfg.get("num_regular_frames", 8),
        start_time=model_cfg.get("start_time", 0),
        end_time=model_cfg.get("end_time", 128),
        tada_start=model_cfg.get("tada_start", 8),
        tada_end=model_cfg.get("tada_end", 12),
        fuse_window=model_cfg.get("fuse_window", 64),
        kernel_sigma=model_cfg.get("kernel_sigma", 5.0),
        apn_learn_mode=model_cfg.get("apn_learn_mode", "window"),
        fuse_window_offset_min=model_cfg.get("fuse_window_offset_min", 0),
        fuse_window_offset_max=model_cfg.get("fuse_window_offset_max", 16),
        fuse_window_offset_init=model_cfg.get("fuse_window_offset_init", 0),
        tau_multiplier_init=model_cfg.get("tau_multiplier_init", 1.0),
        cls_lstm_hidden_dim=model_cfg.get("cls_lstm_hidden_dim", 64),
        cls_lstm_num_layers=model_cfg.get("cls_lstm_num_layers", 1),
        cls_lstm_bidirectional=model_cfg.get("cls_lstm_bidirectional", True),
        cls_lstm_dropout=model_cfg.get("cls_lstm_dropout", 0.0),
        roi_prompt_hidden_dim=model_cfg.get("roi_prompt_hidden_dim", 64),
        roi_prompt_dropout=model_cfg.get("roi_prompt_dropout", 0.1),
        roi_prompt_init_scale=model_cfg.get("roi_prompt_init_scale", 1e-3),
        roi_mask_token_threshold=model_cfg.get("roi_mask_token_threshold", 0.1),
        roi_mask_jitter_enabled=model_cfg.get("roi_mask_jitter_enabled", True),
        roi_mask_jitter_prob=model_cfg.get("roi_mask_jitter_prob", 0.5),
        roi_mask_jitter_ratio=model_cfg.get("roi_mask_jitter_ratio", 0.01),
        background_dropout_enabled=model_cfg.get("background_dropout_enabled", True),
        background_dropout_ratio=model_cfg.get("background_dropout_ratio", 0.8),
        background_dropout_apply_prob=model_cfg.get(
            "background_dropout_apply_prob", 0.5
        ),
        t2_pool_mask_threshold=model_cfg.get("t2_pool_mask_threshold", 0.1),
        t2_pool_apply_mask_jitter=model_cfg.get("t2_pool_apply_mask_jitter", True),
        num_heads=model_cfg.get("num_heads", 4),
        tumor_num_queries=model_cfg.get("tumor_num_queries", 4),
        peri_num_queries=model_cfg.get("peri_num_queries", 2),
        dropout=model_cfg.get("dropout", 0.2),
        rebase_time=model_cfg.get("rebase_time_by_mask_anchor", False),
        time_anchor_valid_index=model_cfg.get("time_anchor_valid_index", 1),
        add_random_interframe_time_delay_enabled=model_cfg.get(
            "add_random_interframe_time_delay", True
        ),
        add_random_global_time_shift_enabled=model_cfg.get(
            "add_random_global_time_shift", True
        ),
        global_time_shift_min_seconds=model_cfg.get(
            "global_time_shift_min_seconds", -45.0
        ),
        global_time_shift_max_seconds=model_cfg.get(
            "global_time_shift_max_seconds", 45.0
        ),
    )


__all__ = [
    "MODEL_NAME",
    "T1_only_OwnMaskTumorPeri_v8_DenseROIPrompt_T2RegionPool",
    "create_model",
]
