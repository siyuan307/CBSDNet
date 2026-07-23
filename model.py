

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. ECA — Efficient Channel Attention
# =========================================================================

class ECA(nn.Module):
    """Efficient Channel Attention (ECA-Net, Wang et al. CVPR 2020).

    Replaces the FC-ReLU-FC bottleneck in SE-Net with a 1D convolution
    of kernel size 3, reducing parameters from 2C²/r to just 3 while
    avoiding the information loss caused by dimensionality reduction.

    Formulation:
        y = Sigmoid( Conv1D( AdaptiveAvgPool(x) ) )
        out = x ⊙ y   (channel-wise multiplication)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = self.avg_pool(x)               # [B, C, 1, 1]
        y = y.squeeze(-1)                   # [B, C, 1]
        y = y.transpose(-1, -2)             # [B, 1, C]
        y = self.conv(y)                    # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y


# =========================================================================
# 2. LCSRBlock — Lightweight Center-Surround Residual Block
# =========================================================================

class LCSRBlock(nn.Module):
    """Lightweight Center-Surround Residual Block.

    Explicitly encodes the center-surround contrast prior from retinal
    ganglion cells and classical IR small-target detection (e.g. Top-Hat).
    Three parallel branches capture complementary information:

      Branch       Kernel    What it captures
      ─────────    ──────    ──────────────────────────────
      Center       3×3 DW    Local bright spot (target itself)
      Surround     7×7 DW    Local background estimate
      Dilate       3×3 DW    Intermediate-scale structure (dilation=2)
      CS diff      c − s     Center-surround contrast ≈ saliency

    The three branches are concatenated, fused via 1×1 conv, then
    passed through ECA channel attention and a residual connection.

    Parameters (C_in=C_out=16): ~2,227
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = norm_type

        # Point-wise projection: C_in → C_out
        self.pw_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Center branch: 3×3 depthwise
        self.dw_center = nn.Conv2d(out_channels, out_channels, 3,
                                   stride=stride, padding=1,
                                   groups=out_channels, bias=False)
        # Surround branch: 7×7 depthwise
        self.dw_surround = nn.Conv2d(out_channels, out_channels, 7,
                                     stride=stride, padding=3,
                                     groups=out_channels, bias=False)
        # Dilated branch: 3×3 depthwise, dilation=2
        self.dw_dilate = nn.Conv2d(out_channels, out_channels, 3,
                                   stride=stride, padding=2, dilation=2,
                                   groups=out_channels, bias=False)

        # Fusion: 3×C_out → C_out
        self.pw_fuse = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)

        # Batch/Group norms
        self.bn_center   = self._norm(out_channels)
        self.bn_surround = self._norm(out_channels)
        self.bn_dilate   = self._norm(out_channels)
        self.bn_fuse     = self._norm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.eca  = ECA(out_channels)

        # Shortcut: identity or 1×1 projection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                self._norm(out_channels),
            )
        else:
            self.shortcut = None

    def _norm(self, channels: int):
        if self.norm_type == 'gn':
            return nn.GroupNorm(min(4, channels), channels)
        return nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        u = self.pw_proj(x)                      # [B, C_out, H, W]

        # Three parallel branches
        c = self.relu(self.bn_center(self.dw_center(u)))       # center
        s = self.relu(self.bn_surround(self.dw_surround(u)))   # surround
        cs = c - s                                              # center-surround diff
        d = self.relu(self.bn_dilate(self.dw_dilate(u)))       # dilated

        # Fuse: concat → 1×1 → BN → ECA → +shortcut → ReLU
        fused = torch.cat([c, cs, d], dim=1)   # [B, 3*C_out, H, W]
        fused = self.pw_fuse(fused)             # [B, C_out, H, W]
        fused = self.bn_fuse(fused)
        fused = self.eca(fused)

        return self.relu(fused + identity)


# =========================================================================
# 3. CBSDM — Continuous Background Spectrum Decoupling Module
# =========================================================================

class CBSDM(nn.Module):
    """Continuous Background Spectrum Decoupling Module.

    This is the core innovation of CBSDNet. It decomposes encoder features
    into background B and target-candidate residual R via a learnable,
    continuous spectral gating function G_b(ρ) in the Fourier domain.

    Key insight: In classical signal processing, background (clouds,
    temperature gradients) manifests as low-frequency components while
    targets (bright spots) appear as high/wide-band components. But a hard
    low-pass cutoff is too crude — CBSDM learns a *continuous* gating
    function parameterized by RBFs, letting the network decide how much
    of each frequency band belongs to the background.

    Pipeline:
        1. Global stats (mean/std/grad per channel) → MLP → RBF weights a, bias b
        2. FFT → X_freq
        3. G_b(ρ) = σ(b + Σₖ aₖ · exp(−(ρ−μₖ)² / 2σₖ²))   [continuous gate]
        4. B = iFFT(G_b ⊙ X_freq),   R = feat − B
        5. C_r = local z-score of R (7×7 window)
        6. P_t = Sigmoid(Conv([|feat|, |R|, C_r]))          [target prior]
        7. P_b = Sigmoid(Conv([|B|, |feat|, |R|]))          [background confidence]

    Parameters (C=16): MLP ~1,152 + pt/pb heads ~896 = ~2,048

    Note: μₖ and σₖ are fixed buffers (uniformly spaced, σ=0.12);
    only aₖ and b are learned through the MLP.
    """
    def __init__(self, in_channels: int, num_rbf: int = 5,
                 norm_type: str = 'bn'):
        super().__init__()
        self.in_channels = in_channels
        self.num_rbf = num_rbf
        self.norm_type = norm_type

        # Fixed RBF centres (uniform over [0.05, 0.95]) and width
        mu = torch.linspace(0.05, 0.95, num_rbf)            # [K]
        sigma = torch.full((num_rbf,), 0.12)                 # [K]
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # MLP: per-channel statistics (3C) → RBF weights (C·K) + bias (C)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 3, max(in_channels // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(in_channels // 2, 8), in_channels * (num_rbf + 1)),
        )

        # P_t head: |feat| + |R| + C_r → target prior probability map
        self.pt_head = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, padding=1, bias=False),
            self._norm(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        # P_b head: |B| + |feat| + |R| → background confidence map
        self.pb_head = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, padding=1, bias=False),
            self._norm(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        self._freq_grid = None  # cached frequency grid

    def _norm(self, channels: int):
        if self.norm_type == 'gn':
            return nn.GroupNorm(min(4, channels), channels)
        return nn.BatchNorm2d(channels)

    def _get_freq_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Build normalised radial frequency grid ρ(u,v) ∈ [0, 1].

        ρ(u,v) = √(f_y(u)² + f_x(v)²) / ρ_max
        where f_y(u) = |u/H| for u ∈ [0, H-1],
              f_x(v) = |v/W| for v ∈ [0, W/2] (rfft half-spectrum).
        """
        if self._freq_grid is not None and self._freq_grid.shape == (H, W // 2 + 1):
            return self._freq_grid.to(device)

        fy = torch.fft.fftfreq(H, device=device).abs().view(-1, 1)     # [H, 1]
        fx = torch.fft.rfftfreq(W, device=device).abs().view(1, -1)    # [1, W/2+1]
        rho = torch.sqrt(fy ** 2 + fx ** 2)                             # [H, W/2+1]
        rho = rho / (rho.max() + 1e-8)
        self._freq_grid = rho
        return rho

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: encoder feature [B, C, H, W]

        Returns:
            B:      background estimate           [B, C, H, W]
            R:      residual (target candidate)   [B, C, H, W]
            P_t:    target prior                  [B, 1, H, W]
            P_b:    background confidence         [B, 1, H, W]
            G_b:    spectral gate                 [B, C, H, W//2+1]
            X_freq: FFT of input                  [B, C, H, W//2+1] (complex)
            C_r:    local residual contrast       [B, C, H, W]
        """
        B, C, H, W = feat.shape

        # --- 1. Global per-channel statistics ---
        F_mean = feat.mean(dim=[2, 3])                                   # [B, C]
        F_std  = feat.std(dim=[2, 3])                                    # [B, C]
        # Mean absolute gradient (horizontal + vertical)
        gx = (feat[:, :, :, 1:] - feat[:, :, :, :-1]).abs().mean(dim=[2, 3])
        gy = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).abs().mean(dim=[2, 3])
        F_grad = (gx + gy) / 2

        stats = torch.cat([F_mean, F_std, F_grad], dim=1)                # [B, 3C]
        params = self.mlp(stats)                                          # [B, C·(K+1)]
        a = params[:, :C * self.num_rbf].view(B, C, self.num_rbf)        # [B, C, K]
        b = params[:, C * self.num_rbf:].view(B, C, 1)                   # [B, C, 1]

        # --- 2. FFT to frequency domain ---
        X_freq = torch.fft.rfft2(feat)                                    # [B, C, H, W/2+1]

        # --- 3. Build continuous spectral gate G_b(ρ) ---
        rho = self._get_freq_grid(H, W, feat.device)                      # [H, W/2+1]

        # RBF basis: φ_k(ρ) = exp(−(ρ − μ_k)² / (2 σ_k²))
        phi = torch.exp(
            -(rho.unsqueeze(0) - self.mu.view(-1, 1, 1)) ** 2
            / (2 * self.sigma.view(-1, 1, 1) ** 2)
        )                                                                 # [K, H, W/2+1]
        phi = phi.unsqueeze(0)                                            # [1, K, H, W/2+1]

        # Weighted sum: Σ_k a_{c,k} · φ_k(ρ)
        sum_aphi = (a.unsqueeze(-1).unsqueeze(-1) * phi.unsqueeze(1)).sum(dim=2)
        # G_b(ρ) = σ(b + Σ a_k · φ_k(ρ))
        G_b = torch.sigmoid(b.unsqueeze(-1) + sum_aphi)                  # [B, C, H, W/2+1]

        # --- 4. Background B and residual R ---
        B = torch.fft.irfft2(G_b * X_freq, s=(H, W))                     # [B, C, H, W]
        R = feat - B                                                       # [B, C, H, W]

        # --- 5. Local residual contrast C_r (7×7 z-score) ---
        k = 7
        R_avg  = F.avg_pool2d(R, k, stride=1, padding=k // 2)
        R2_avg = F.avg_pool2d(R ** 2, k, stride=1, padding=k // 2)
        C_r = (R - R_avg) / torch.sqrt(R2_avg + 1e-6)                   # [B, C, H, W]

        # --- 6. Generate target prior P_t and background confidence P_b ---
        R_abs   = R.abs().mean(dim=1, keepdim=True)                      # [B, 1, H, W]
        Cr_mean = C_r.mean(dim=1, keepdim=True)                           # [B, 1, H, W]
        pt_in = torch.cat([
            feat.abs().mean(dim=1, keepdim=True), R_abs, Cr_mean
        ], dim=1)                                                        # [B, 3, H, W]

        B_abs = B.abs().mean(dim=1, keepdim=True)
        pb_in = torch.cat([
            B_abs, feat.abs().mean(dim=1, keepdim=True), R_abs
        ], dim=1)                                                        # [B, 3, H, W]

        P_t = self.pt_head(pt_in)                                        # [B, 1, H, W]
        P_b = self.pb_head(pb_in)                                        # [B, 1, H, W]

        return B, R, P_t, P_b, G_b, X_freq, C_r


# =========================================================================
# 4. BRCM — Background-Residual Coupled Modulation
# =========================================================================

class BRCM(nn.Module):
    """Background-Residual Coupled Modulation.

    Uses the spatial priors P_t/P_b from CBSDM to directly modulate
    encoder features — enhancing target regions and suppressing background
    regions — *without* an intermediate gating network.

    Formulation:
        X_out = X + α · P_t · E_s                ← target enhancement
        X_out = X_out · (1 − β · P_b) + δ · PWConv(X)  ← bg suppression + bypass

    where:
        E_s = DWConv_{3×3}(X)    local refinement feature
        α, β, δ                  learnable scalars

    Key design choice: P_t/P_b directly multiply the modulation terms.
    Removing the gating CNN prevents the network from learning to "switch
    off" the modulation path (a pathology observed in earlier versions
    where α/β would decay to zero).

    Initialisation: α=0.01, β=0.05, δ=0 → starts as identity, gradually
    learns to modulate. β needs a larger initial value because the
    multiplicative suppression (1−β·P_b) generates stronger gradients.

    Emergent behaviour: Across the three encoder stages, the network
    spontaneously specialises:
      Level 0 (256×256): little modulation (targets tiny, easy to damage)
      Level 1 (128×128): mainly target enhancement
      Level 2  (64×64):  mainly background suppression

    Parameters (C=16): DWConv 144 + PWConv 256 + α,β,δ 3 = ~403
    """
    def __init__(self, channels: int, norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = norm_type

        # Depthwise conv for local refinement E_s
        self.dsconv = nn.Conv2d(channels, channels, 3, padding=1,
                                groups=channels, bias=False)
        # Pointwise conv for residual bypass path
        self.pwconv = nn.Conv2d(channels, channels, 1, bias=False)

        # Learnable modulation strengths
        self.alpha = nn.Parameter(torch.tensor(0.01))   # target enhancement
        self.beta  = nn.Parameter(torch.tensor(0.05))   # background suppression
        self.delta = nn.Parameter(torch.zeros(1))       # residual bypass (zero-init)

    def forward(self, X_s: torch.Tensor,
                P_t: torch.Tensor, P_b: torch.Tensor):
        """
        Args:
            X_s: encoder feature [B, C, H, W]
            P_t: target prior    [B, 1, H', W'] (from CBSDM)
            P_b: background conf [B, 1, H', W'] (from CBSDM)

        Returns:
            X_out: modulated feature [B, C, H, W]
            E_s:   local refinement  [B, C, H, W] (for diagnostics)
        """
        # Align P_t/P_b to feature spatial size
        pt = F.interpolate(P_t, size=X_s.shape[2:],
                           mode='bilinear', align_corners=False)
        pb = F.interpolate(P_b, size=X_s.shape[2:],
                           mode='bilinear', align_corners=False)

        E_s = self.dsconv(X_s)

        # Target enhancement: X ← X + α·P_t·E_s
        X_out = X_s + self.alpha * pt * E_s
        # Background suppression + residual bypass
        X_out = X_out * (1.0 - self.beta * pb) + self.delta * self.pwconv(X_s)

        return X_out, E_s


# =========================================================================
# 5. TPB — Target Preservation Block
# =========================================================================

class TPB(nn.Module):
    """Target Preservation Block.

    Addresses the inherent boundary blurring caused by repeated
    downsampling–upsampling in the encoder–decoder path. TPB fuses the
    stem feature f0 (which retains full-resolution spatial detail after
    only one 1×1 conv) with the decoder output d0, guided by the CBSDM
    target prior P_t.

    Two-stage refinement:
      Stage 1 — Detail Injection:
        D_t = ConvBlock(P_t)                    [1 → C channels]
        (P_t tells "where to look" for detail)

      Stage 2 — Feature Refinement:
        D_refined = ConvBlock([d0, D_t, P_t])   [2C+1 → C channels]
        (semantic context + injected detail + spatial guidance)

    Parameters: ~2,500
    """
    def __init__(self, in_channels: int, norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = norm_type

        # Stage 1: Detail injection from P_t
        self.conv_dt = nn.Sequential(
            nn.Conv2d(1, in_channels, 3, padding=1, bias=False),
            self._norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),   # depthwise
            self._norm(in_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Feature refinement
        self.conv_refine = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels, 3, padding=1, bias=False),
            self._norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),      # 1×1 linear projection
        )

    def _norm(self, channels: int):
        if self.norm_type == 'gn':
            return nn.GroupNorm(min(4, channels), channels)
        return nn.BatchNorm2d(channels)

    def forward(self, D_dec: torch.Tensor, P_t: torch.Tensor):
        """
        Args:
            D_dec: decoder feature [B, C, H, W]
            P_t:   target prior   [B, 1, H, W]

        Returns:
            D_t:        injected detail feature  [B, C, H, W]
            D_refined:  refined feature           [B, C, H, W]
        """
        D_t = self.conv_dt(P_t)                                 # [B, C, H, W]
        refine_in = torch.cat([D_dec, D_t, P_t], dim=1)         # [B, 2C+1, H, W]
        D_refined = self.conv_refine(refine_in)                  # [B, C, H, W]
        return D_t, D_refined


# =========================================================================
# 6. BoundaryHead — Boundary Prediction Head
# =========================================================================

class BoundaryHead(nn.Module):
    """Boundary prediction head — auxiliary training supervision only.

    Predicts the boundary mask E_pred from the same inputs as TPB.
    The boundary ground truth is generated by morphological operations:
        E_gt = dilate(mask, 3×3) \ erode(mask, 3×3)

    This head is used only during training to provide boundary-aware
    regularisation; it is not used at inference time.

    Parameters: ~2,000
    """
    def __init__(self, in_channels: int, norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels, 3, padding=1, bias=False),
            self._norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def _norm(self, channels: int):
        if self.norm_type == 'gn':
            return nn.GroupNorm(min(4, channels), channels)
        return nn.BatchNorm2d(channels)

    def forward(self, D_dec: torch.Tensor, D_t: torch.Tensor,
                P_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            D_dec: decoder feature [B, C, H, W]
            D_t:   TPB detail     [B, C, H, W]
            P_t:   target prior   [B, 1, H, W]

        Returns:
            E_pred: boundary prediction logits [B, 1, H, W]
        """
        x = torch.cat([D_dec, D_t, P_t], dim=1)
        return self.conv(x)


# =========================================================================
# 7. CBSDNet — Full Model
# =========================================================================

class CBSDNet(nn.Module):
    """CBSDNet: Continuous Background Spectrum Decoupling Network.

    A lightweight (0.74 M parameters) encoder-decoder network for infrared
    small target detection. The architecture follows a 2+1+2 layout:

      Encoder (2 stages):  LCSRBlock×2 → CBSDM → BRCM  per stage
      Bottleneck (1 stage): LCSRBlock×2
      Decoder (2 stages):   skip connection + LCSRBlock×2  per stage
      TPB:                  detail injection + refinement at full resolution
      BoundaryHead:         auxiliary boundary supervision

    Channel progression:
      Level 0: 16 ch @ H×W       Level 1: 32 ch @ H/2×W/2
      Bottleneck: 128 ch @ H/4×W/4

    The forward pass returns a rich dictionary containing:
      - mask, main_mask, tpb_mask:     predictions
      - B, R, Pt, Pb, G_b:             CBSDM outputs (per stage)
      - e, e_mod:                       encoder features before/after BRCM
      - boundary:                       auxiliary boundary prediction

    Args:
        input_channels: 1 for grayscale IR, 3 for RGB/3-band
        norm_type:      'bn' (BatchNorm) or 'gn' (GroupNorm)
    """
    def __init__(self, input_channels: int = 3,
                 norm_type: str = 'bn'):
        super().__init__()

        # Channel configuration: [enc_0, enc_1, bottleneck]
        ch = [16, 32, 128]

        # --- Stem ---
        self.conv_init = nn.Conv2d(input_channels, ch[0], 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Encoder (2 stages, 2 LCSRBlocks each) ---
        self.encoder_0 = nn.Sequential(
            LCSRBlock(ch[0], ch[0], norm_type=norm_type),
            LCSRBlock(ch[0], ch[0], norm_type=norm_type),
        )
        self.encoder_1 = nn.Sequential(
            LCSRBlock(ch[0], ch[1], norm_type=norm_type),
            LCSRBlock(ch[1], ch[1], norm_type=norm_type),
        )

        # --- Bottleneck (2 LCSRBlocks, 128 ch @ H/4×W/4) ---
        self.middle = nn.Sequential(
            LCSRBlock(ch[1], ch[2], norm_type=norm_type),
            LCSRBlock(ch[2], ch[2], norm_type=norm_type),
        )

        # --- CBSDM modules (stage 0 and stage 1) ---
        self.cbsdm_0 = CBSDM(ch[0], norm_type=norm_type)
        self.cbsdm_1 = CBSDM(ch[1], norm_type=norm_type)

        # --- BRCM modules (stage 0 and stage 1) ---
        self.brcm_0 = BRCM(ch[0], norm_type=norm_type)
        self.brcm_1 = BRCM(ch[1], norm_type=norm_type)

        # --- Decoder (2 stages, 2 LCSRBlocks each) ---
        # Decoder 1: up(m) [128] + e1_mod [32] = 160 → 32
        self.decoder_1 = nn.Sequential(
            LCSRBlock(ch[1] + ch[2], ch[1], norm_type=norm_type),
            LCSRBlock(ch[1], ch[1], norm_type=norm_type),
        )
        # Decoder 0: up(d1) [32] + e0_mod [16] + P_t [1] = 49 → 16
        self.decoder_0 = nn.Sequential(
            LCSRBlock(ch[0] + ch[1] + 1, ch[0], norm_type=norm_type),
            LCSRBlock(ch[0], ch[0], norm_type=norm_type),
        )

        # --- Output heads ---
        self.output_0 = nn.Conv2d(ch[0], 1, 1)     # main prediction
        self.output_1 = nn.Conv2d(ch[1], 1, 1)     # auxiliary (warm-up only)
        self.final    = nn.Conv2d(2, 1, 3, 1, 1)   # fuse multi-scale

        # --- TPB + BoundaryHead ---
        self.tpb = TPB(ch[0], norm_type=norm_type)
        _norm = (nn.GroupNorm(min(4, ch[0]), ch[0]) if norm_type == 'gn'
                 else nn.BatchNorm2d(ch[0]))
        self.output_tpb = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], 3, padding=1, bias=False),
            _norm,
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], 1, 1),
        )
        self.boundary_head = BoundaryHead(ch[0], norm_type=norm_type)

    def _run_stage(self, stage_idx: int, init_feat: torch.Tensor,
                   prev_mod: torch.Tensor):
        """Run one encoder stage: LCSRBlocks → CBSDM → BRCM.

        Args:
            stage_idx: 0 or 1
            init_feat: the stem feature f0 (for consistent interface)
            prev_mod:  modulated feature from previous stage (or f0 for stage 0)

        Returns:
            e:      raw encoder feature
            e_mod:  BRCM-modulated feature
            b, r, pt, pb, gb, x_freq, cr: CBSDM outputs
            e_s:    BRCM local refinement feature
        """
        encoders = [self.encoder_0, self.encoder_1]
        cbsdms   = [self.cbsdm_0, self.cbsdm_1]
        brcms    = [self.brcm_0, self.brcm_1]

        encoder = encoders[stage_idx]
        cbsdm   = cbsdms[stage_idx]
        brcm    = brcms[stage_idx]

        # Downsample before stage 1+
        if stage_idx > 0:
            e = encoder(self.pool(prev_mod))
        else:
            e = encoder(init_feat)

        # CBSDM: frequency-domain background/residual decomposition
        b, r, pt, pb, gb, x_freq, cr = cbsdm(e)

        # BRCM: prior-driven spatial modulation
        e_mod, e_s = brcm(e, pt, pb)

        return e, e_mod, b, r, pt, pb, gb, x_freq, cr, e_s

    def forward(self, x: torch.Tensor, warm_flag: bool = True):
        """
        Args:
            x:         input image   [B, C_in, H, W]
            warm_flag: if True, produce multi-scale auxiliary outputs
                       (used during warm-up training epochs)

        Returns:
            dict with keys:
                mask:        final prediction          [B, 1, H, W]
                main_mask:   main branch prediction    [B, 1, H, W]
                tpb_mask:    TPB-refined prediction    [B, 1, H, W]
                boundary:    boundary prediction       [B, 1, H, W]
                masks:       multi-scale aux outputs   list of [B, 1, *, *]
                B, R:        CBSDM B/R per stage       list of [B, C, *, *]
                Pt, Pb:      CBSDM priors per stage    list of [B, 1, *, *]
                Gb:          spectral gates per stage  list of [B, C, *, *]
                e, e_mod:    encoder features          list of [B, C, *, *]
                d:           decoder features          list of [B, C, *, *]
                D_t, D_refined: TPB internal features
        """
        # --- Stem ---
        f0 = self.conv_init(x)                               # [B, 16, H, W]

        # --- Encoder stages ---
        e0, e0_mod, b0, r0, pt0, pb0, gb0, xf0, cr0, es0 = \
            self._run_stage(0, f0, f0)
        e1, e1_mod, b1, r1, pt1, pb1, gb1, xf1, cr1, es1 = \
            self._run_stage(1, f0, e0_mod)

        # --- Bottleneck ---
        m = self.middle(self.pool(e1_mod))                    # [B, 128, H/4, W/4]

        # --- Decoder ---
        d1 = self.decoder_1(torch.cat([e1_mod, self.up(m)], dim=1))
        # Decoder 0 receives P_t as an extra spatial guidance channel
        d0 = self.decoder_0(
            torch.cat([e0_mod, self.up(d1), pt0], dim=1)
        )                                                    # [B, 16, H, W]

        # --- Main prediction ---
        main_pred = self.output_0(d0)                         # [B, 1, H, W]

        # --- TPB: target-preserving boundary refinement ---
        D_t, D_refined = self.tpb(d0.detach(), pt0.detach())
        tpb_pred = self.output_tpb(D_refined)                 # [B, 1, H, W]

        # --- Boundary auxiliary prediction ---
        E_pred = self.boundary_head(d0, D_t.detach(), pt0.detach())

        # --- Multi-scale warm-up outputs ---
        if warm_flag:
            mask0 = main_pred
            mask1 = self.output_1(d1)
            fused = self.final(torch.cat([
                mask0,
                F.interpolate(mask1, size=mask0.shape[2:],
                              mode='bilinear', align_corners=True)
            ], dim=1))
            masks_list = [mask0, mask1]
        else:
            masks_list = []
            fused = main_pred

        return {
            # Predictions
            'mask':       fused,                    # final output (warm-up: fused; else: main)
            'main_mask':  main_pred,                # main branch prediction
            'tpb_mask':   tpb_pred,                 # TPB-refined prediction (used at inference)
            'boundary':   E_pred,                   # boundary auxiliary
            'masks':      masks_list,               # multi-scale aux outputs

            # CBSDM outputs (per stage)
            'B':   [b0, b1],                        # background estimates
            'R':   [r0, r1],                        # residuals (target candidates)
            'Pt':  [pt0, pt1],                      # target priors
            'Pb':  [pb0, pb1],                      # background confidences
            'Gb':  [gb0, gb1],                      # spectral gates
            'X_freq': [xf0, xf1],                   # FFT of encoder features
            'C_r': [cr0, cr1],                      # local residual contrasts

            # Encoder features (for loss computation)
            'e':     [e0, e1],                      # raw encoder features
            'e_mod': [e0_mod, e1_mod],              # BRCM-modulated features
            'E_s':   [es0, es1],                    # BRCM local refinement

            # Decoder features
            'd': [d0, d1],

            # TPB internals
            'D_t':        D_t,                       # injected detail
            'D_refined':  D_refined,                 # refined feature
        }


# =========================================================================
# Quick test
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CBSDNet — Model Component Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBSDNet(input_channels=3, norm_type='bn').to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dummy forward
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(x, warm_flag=True)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output keys:  {list(out.keys())}")
    print(f"  mask:        {out['mask'].shape}")
    print(f"  main_mask:   {out['main_mask'].shape}")
    print(f"  tpb_mask:    {out['tpb_mask'].shape}")
    print(f"  boundary:    {out['boundary'].shape}")
    print(f"  B[0]:        {out['B'][0].shape}")
    print(f"  R[0]:        {out['R'][0].shape}")
    print(f"  Pt[0]:       {out['Pt'][0].shape}")
    print(f"  Gb[0]:       {out['Gb'][0].shape}")
    print(f"  D_t:         {out['D_t'].shape}")
    print(f"  D_refined:   {out['D_refined'].shape}")
    print(f"\nAll tests passed! ✓")
