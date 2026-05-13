"""
Baseline Loss Functions for MM-LeJEPA Experiments.

Implements standard self-supervised learning losses used as baselines:
1. VICReg (Variance-Invariance-Covariance Regularization)
2. InfoNCE (Noise-Contrastive Estimation)
3. AdaSigNCE (Adaptive SigReg-guided InfoNCE)

These replace the LeJEPA SIGReg objective in ablation studies.

References:
- VICReg: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning", ICLR 2022.
- InfoNCE: Oord et al., "Representation Learning with Contrastive Predictive Coding",
  arXiv:1807.03748.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _flatten_views(proj: torch.Tensor) -> torch.Tensor:
    if proj.dim() == 2:
        return proj
    if proj.dim() != 3:
        raise ValueError(f"Expected proj with 2 or 3 dims, got shape {tuple(proj.shape)}")
    return proj.reshape(-1, proj.shape[-1])


def _safe_eigh(cov: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    cov = cov.float()
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov + eye * eps)
    eigvals = eigvals.clamp_min(eps)
    return eigvals, eigvecs


def _whiten_tensor(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    orig_dtype = z.dtype
    z_centered = (z - z.mean(dim=0, keepdim=True)).float()
    cov = (z_centered.T @ z_centered) / max(z_centered.shape[0] - 1, 1)
    eigvals, eigvecs = _safe_eigh(cov, eps)
    whitener = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return (z_centered @ whitener).to(orig_dtype)


def _build_logits(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    batch_size = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = (z @ z.T) / temperature
    mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=sim.device),
        torch.arange(0, batch_size, device=sim.device),
    ])
    return sim, labels


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def _dist_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        dist.all_reduce(tensor)
    return tensor


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    Three terms:
      - Invariance: MSE between embeddings of different views.
      - Variance: Hinge loss to keep per-dimension std above a threshold.
      - Covariance: Penalise off-diagonal elements of the covariance matrix
        to decorrelate embedding dimensions.

    Operates on projected embeddings shaped (V, B, D) where V is views.
    The invariance loss is computed between all pairs of views (or the
    global-center strategy used by the rest of this codebase).

    Args:
        sim_coeff: Weight for invariance term (default 25.0).
        std_coeff: Weight for variance term (default 25.0).
        cov_coeff: Weight for covariance term (default 1.0).
        std_target: Target standard deviation for hinge loss (default 1.0).
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        std_target: float = 1.0,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_target = std_target

    # ------------------------------------------------------------------
    @staticmethod
    def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        """Return flattened off-diagonal elements of a square matrix."""
        n = matrix.shape[0]
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # ------------------------------------------------------------------
    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss on per-dim std: max(0, target - std(z_d))."""
        std = z.std(dim=0)
        return F.relu(self.std_target - std).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Encourage off-diagonal covariance to be zero."""
        N, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(N - 1, 1)
        return self._off_diagonal(cov).pow(2).sum() / D

    # ------------------------------------------------------------------
    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (V, B, D) projected embeddings.

        Returns:
            Scalar loss.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # (1, B, D)

        V, B, D = proj.shape

        # --- Invariance (MSE between view pairs) ---
        # Use global-center strategy consistent with LeJEPA codebase:
        # center = mean of global views (first V_g), pull every view to center.
        center = proj.mean(0)  # (B, D)
        inv_loss = (proj - center.unsqueeze(0)).pow(2).mean()

        # --- Variance & Covariance (per-view, averaged) ---
        var_loss = torch.tensor(0.0, device=proj.device)
        cov_loss = torch.tensor(0.0, device=proj.device)
        for v in range(V):
            z = proj[v]  # (B, D)
            var_loss = var_loss + self.variance_loss(z)
            cov_loss = cov_loss + self.covariance_loss(z)
        var_loss = var_loss / V
        cov_loss = cov_loss / V

        loss = (
            self.sim_coeff * inv_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def forward_components(self, proj: torch.Tensor):
        """Return individual loss terms for logging."""
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        V, B, D = proj.shape
        center = proj.mean(0)
        inv_loss = (proj - center.unsqueeze(0)).pow(2).mean()
        var_loss = torch.tensor(0.0, device=proj.device)
        cov_loss = torch.tensor(0.0, device=proj.device)
        for v in range(V):
            z = proj[v]
            var_loss = var_loss + self.variance_loss(z)
            cov_loss = cov_loss + self.covariance_loss(z)
        var_loss = var_loss / V
        cov_loss = cov_loss / V
        return {
            'invariance': inv_loss,
            'variance': var_loss,
            'covariance': cov_loss,
            'total': self.sim_coeff * inv_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss,
        }


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent style) contrastive loss.

    For each view pair, treats same-sample representations as positives
    and all other samples in the batch as negatives.

    Operates on projected embeddings shaped (V, B, D).

    When V >= 2 the loss is computed between all adjacent view pairs
    (view 0 vs view 1, view 1 vs view 2, …).  When V == 1 it expects
    ``proj`` to already be stacked as (2, B, D) from two augmented views.

    Args:
        temperature: Softmax temperature (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _pairwise_nce(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE between two batches (B, D)."""
        sim, labels = _build_logits(z_a, z_b, self.temperature)
        loss = F.cross_entropy(sim, labels)
        return loss

    # ------------------------------------------------------------------
    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (V, B, D) projected embeddings.

        Returns:
            Scalar loss.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        V, B, D = proj.shape
        if V < 2:
            # Need at least 2 views – return zero gracefully
            return torch.tensor(0.0, device=proj.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=proj.device)
        n_pairs = 0
        for i in range(V):
            for j in range(i + 1, V):
                total_loss = total_loss + self._pairwise_nce(proj[i], proj[j])
                n_pairs += 1
        return total_loss / max(n_pairs, 1)

    def forward_components(self, proj: torch.Tensor):
        """Return loss dict for logging compatibility."""
        loss = self.forward(proj)
        return {'infonce': loss, 'total': loss}


class WhitenedInfoNCELoss(InfoNCELoss):
    """InfoNCE after batch whitening to reduce anisotropy-driven shortcuts."""

    def __init__(self, temperature: float = 0.07, whiten_eps: float = 1e-4):
        super().__init__(temperature=temperature)
        self.whiten_eps = whiten_eps

    def _pairwise_nce(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([z_a, z_b], dim=0)
        joint_white = _whiten_tensor(joint, eps=self.whiten_eps)
        z_a_white, z_b_white = joint_white[:z_a.shape[0]], joint_white[z_a.shape[0]:]
        sim, labels = _build_logits(z_a_white, z_b_white, self.temperature)
        return F.cross_entropy(sim, labels)

    def forward_components(self, proj: torch.Tensor):
        loss = self.forward(proj)
        flat = _flatten_views(proj)
        with torch.no_grad():
            z_white = _whiten_tensor(flat, eps=self.whiten_eps)
            cov = (z_white.T @ z_white) / max(z_white.shape[0] - 1, 1)
            whitening_offdiag = VICRegLoss._off_diagonal(cov).pow(2).mean()
        return {
            'whitened_infonce': loss,
            'whitening_offdiag': whitening_offdiag,
            'total': loss,
        }


class EigenNCELoss(InfoNCELoss):
    """InfoNCE in a covariance-eigenspace with spectrum-aware reweighting."""

    def __init__(
        self,
        temperature: float = 0.07,
        eig_power: float = 0.5,
        eps: float = 1e-4,
    ):
        super().__init__(temperature=temperature)
        self.eig_power = eig_power
        self.eps = eps

    def _eigen_reweight(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = z_a.dtype
        joint = torch.cat([z_a, z_b], dim=0)
        joint_centered = (joint - joint.mean(dim=0, keepdim=True)).float()
        cov = (joint_centered.T @ joint_centered) / max(joint_centered.shape[0] - 1, 1)
        eigvals, eigvecs = _safe_eigh(cov, self.eps)
        weights = eigvals.rsqrt().pow(self.eig_power)
        projected = joint_centered @ eigvecs
        projected = projected * weights.unsqueeze(0)
        projected = projected.to(orig_dtype)
        return projected[:z_a.shape[0]], projected[z_a.shape[0]:], eigvals

    def _pairwise_nce(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a_eig, z_b_eig, _ = self._eigen_reweight(z_a, z_b)
        sim, labels = _build_logits(z_a_eig, z_b_eig, self.temperature)
        return F.cross_entropy(sim, labels)

    def forward_components(self, proj: torch.Tensor):
        loss = self.forward(proj)
        flat = _flatten_views(proj)
        half = max(flat.shape[0] // 2, 1)
        z_a = flat[:half]
        z_b = flat[half:half + half]
        if z_b.shape[0] != z_a.shape[0]:
            z_b = flat[-z_a.shape[0]:]
        with torch.no_grad():
            _, _, eigvals = self._eigen_reweight(z_a, z_b)
        return {
            'eigennce': loss,
            'eig_top': eigvals[-1],
            'eig_bottom': eigvals[0],
            'eig_condition': eigvals[-1] / eigvals[0].clamp_min(self.eps),
            'total': loss,
        }


class DINOHead(nn.Module):
    """DINOv3-style projection head without the older weight-normalized classifier."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True,
        use_bn: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()
        nlayers = max(int(nlayers), 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim, bias=mlp_bias)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=mlp_bias)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=mlp_bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=mlp_bias))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self._init_weights(self.last_layer)
        if norm_last_layer:
            self.last_layer.weight.requires_grad = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        no_last_layer: bool = False,
        only_last_layer: bool = False,
    ) -> torch.Tensor:
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = F.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regularizer used in DINOv3."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.pdist = nn.PairwiseDistance(2, eps=eps)

    def pairwise_nns_inner(self, x: torch.Tensor) -> torch.Tensor:
        sim = x @ x.T
        sim.fill_diagonal_(-1.0)
        return sim.argmax(dim=1)

    def forward(self, student_output: torch.Tensor) -> torch.Tensor:
        if student_output.numel() == 0:
            return student_output.new_tensor(0.0)
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output.float(), dim=-1, p=2, eps=self.eps)
            indices = self.pairwise_nns_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])
            return -torch.log(distances + self.eps).mean()


class DINOLoss(nn.Module):
    """DINO-style cross-view self-distillation with official global/local loss semantics."""

    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp: float = 0.07,
        teacher_temp_warmup_start: float = 0.04,
        teacher_temp_warmup_epochs: int = 30,
        center_momentum: float = 0.9,
        use_sinkhorn_teacher: bool = False,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_start = teacher_temp_warmup_start
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.center_momentum = center_momentum
        self.use_sinkhorn_teacher = use_sinkhorn_teacher
        self.sinkhorn_iterations = sinkhorn_iterations
        self.register_buffer('center', torch.zeros(1, 1))

    def _ensure_center_shape(self, dim: int, device: torch.device, dtype: torch.dtype):
        if self.center.shape[-1] != dim:
            self.center = torch.zeros(1, dim, device=device, dtype=dtype)

    def get_teacher_temp(self, epoch: int, total_epochs: int) -> float:
        warmup_epochs = min(max(self.teacher_temp_warmup_epochs, 0), max(total_epochs, 1))
        if warmup_epochs <= 1:
            return self.teacher_temp
        if epoch >= warmup_epochs:
            return self.teacher_temp
        alpha = epoch / max(warmup_epochs - 1, 1)
        return self.teacher_temp_warmup_start + alpha * (self.teacher_temp - self.teacher_temp_warmup_start)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor, teacher_temp: float, n_iterations: int | None = None) -> torch.Tensor:
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).T
        batch_size = torch.tensor(float(Q.shape[1]), device=Q.device, dtype=Q.dtype)
        _dist_all_reduce_sum(batch_size)
        B = batch_size.clamp_min(1.0)
        K = Q.shape[0]
        sum_Q = Q.sum()
        _dist_all_reduce_sum(sum_Q)
        Q /= sum_Q.clamp_min(1e-12)
        n_iterations = self.sinkhorn_iterations if n_iterations is None else n_iterations
        for _ in range(n_iterations):
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            _dist_all_reduce_sum(sum_of_rows)
            Q /= sum_of_rows.clamp_min(1e-12)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            Q /= B
        Q *= B
        return Q.T

    @torch.no_grad()
    def get_teacher_probs(
        self,
        teacher_proj: torch.Tensor,
        teacher_global_views: int = 2,
        teacher_temp: float | None = None,
    ) -> torch.Tensor:
        if teacher_proj.dim() != 3:
            raise ValueError('Teacher projections must have shape (V, B, D)')

        teacher_proj = teacher_proj.float()
        teacher_global_views = min(max(teacher_global_views, 1), teacher_proj.shape[0])
        teacher_temp = self.teacher_temp if teacher_temp is None else teacher_temp
        teacher_output = teacher_proj[:teacher_global_views].reshape(-1, teacher_proj.shape[-1])
        self._ensure_center_shape(teacher_output.shape[-1], teacher_output.device, teacher_output.dtype)

        if self.use_sinkhorn_teacher:
            teacher_probs = self.sinkhorn_knopp_teacher(teacher_output, teacher_temp=teacher_temp)
        else:
            teacher_center = self.center.to(device=teacher_output.device, dtype=teacher_output.dtype)
            teacher_probs = F.softmax((teacher_output - teacher_center) / teacher_temp, dim=-1)
            batch_center = teacher_output.sum(dim=0, keepdim=True)
            batch_count = torch.tensor(float(teacher_output.shape[0]), device=teacher_output.device, dtype=teacher_output.dtype)
            _dist_all_reduce_sum(batch_center)
            _dist_all_reduce_sum(batch_count)
            batch_center = batch_center / batch_count.clamp_min(1.0)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

        return teacher_probs.reshape(teacher_global_views, teacher_proj.shape[1], -1).detach()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        ignore_diagonal: bool = False,
    ) -> torch.Tensor:
        if student_logits.dim() != 3 or teacher_probs.dim() != 3:
            raise ValueError('DINO loss expects (V, B, D) tensors')
        if student_logits.shape[1] != teacher_probs.shape[1] or student_logits.shape[2] != teacher_probs.shape[2]:
            raise ValueError(
                f'Mismatched DINO tensors: student={tuple(student_logits.shape)}, teacher={tuple(teacher_probs.shape)}'
            )
        if student_logits.shape[0] == 0 or teacher_probs.shape[0] == 0:
            return student_logits.new_tensor(0.0)

        student_log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        batch_size = student_logits.shape[1]
        student_crops = student_logits.shape[0]
        teacher_crops = teacher_probs.shape[0]

        if not ignore_diagonal:
            loss = -torch.einsum('s b k, t b k ->', student_log_probs, teacher_probs)
            return loss / max(batch_size * student_crops * teacher_crops, 1)

        loss = -torch.einsum('s b k, t b k -> s t', student_log_probs, teacher_probs)
        diagonal_terms = min(student_crops, teacher_crops)
        loss = torch.diagonal_scatter(loss, loss.new_zeros(diagonal_terms))
        denom = batch_size * student_crops * teacher_crops - batch_size * diagonal_terms
        if denom <= 0:
            return loss.new_tensor(0.0)
        return loss.sum() / denom


class IBOTPatchLoss(nn.Module):
    """Simplified RGB-only iBOT-style patch distillation loss for aggressive DINO baselines."""

    def __init__(
        self,
        student_temp: float = 0.1,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn_iterations = sinkhorn_iterations

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).T
        batch_size = torch.tensor(float(Q.shape[1]), device=Q.device, dtype=Q.dtype)
        _dist_all_reduce_sum(batch_size)
        B = batch_size.clamp_min(1.0)
        K = Q.shape[0]
        sum_Q = Q.sum()
        _dist_all_reduce_sum(sum_Q)
        Q /= sum_Q.clamp_min(1e-12)
        for _ in range(self.sinkhorn_iterations):
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            _dist_all_reduce_sum(sum_of_rows)
            Q /= sum_of_rows.clamp_min(1e-12)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            Q /= B
        Q *= B
        return Q.T

    def forward_masked(
        self,
        student_patch_tokens_masked: torch.Tensor,
        teacher_patch_tokens_masked: torch.Tensor,
        teacher_temp: float,
    ) -> torch.Tensor:
        if student_patch_tokens_masked.numel() == 0 or teacher_patch_tokens_masked.numel() == 0:
            return student_patch_tokens_masked.new_tensor(0.0)
        teacher_probs = self.sinkhorn_knopp_teacher(teacher_patch_tokens_masked, teacher_temp=teacher_temp).detach()
        student_log_probs = F.log_softmax(student_patch_tokens_masked.float() / self.student_temp, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


class AdaSigNCELoss(nn.Module):
    """Adaptive Spectral-Contrastive Learning (AdaSigNCE).

    Combines InfoNCE contrastive signal with SIGReg spectral regularization
    through a *functional* coupling rather than a naive weighted sum.

    Core idea: SIGReg's characteristic-function test statistic measures
    how far the current embedding distribution is from Gaussian (i.e. from
    collapse).  We use this statistic to:

    1. **Adaptive temperature**: When the distribution is unhealthy (high
       SIGReg statistic → collapsing), soften the InfoNCE temperature to
       encourage exploration.  When healthy, sharpen it for fine-grained
       discrimination.
    2. **Adaptive weighting**: The SIGReg regularization weight increases
       automatically when collapse risk is higher, creating a self-healing
       feedback loop.
    3. **Dual-space contrastive**: InfoNCE operates in L2-normalized space
       (cosine similarity) while SIGReg operates on un-normalized projections.
       The two losses provide complementary gradient signals — angular
       alignment from InfoNCE and marginal uniformity from SIGReg.

    Properties:
    - Self-regulating: no manual temperature tuning needed.
    - Small-batch friendly: SIGReg prevents collapse even with few negatives.
    - Theoretically motivated: connects spectral distribution testing to
      contrastive learning temperature scheduling.

    Args:
        tau_base: Base InfoNCE temperature (default 0.07).
        tau_max: Maximum temperature when collapse is detected (default 0.5).
        sigreg_weight: Base weight for SIGReg component (default 0.5).
        adaptive_tau: Whether to use adaptive temperature (default True).
            When False, falls back to fixed temperature — useful as ablation.
        knots: SIGReg characteristic function integration knots (default 17).
    """

    def __init__(
        self,
        tau_base: float = 0.07,
        tau_max: float = 0.5,
        sigreg_weight: float = 0.5,
        adaptive_tau: bool = True,
        knots: int = 17,
    ):
        super().__init__()
        self.tau_base = tau_base
        self.tau_max = tau_max
        self.sigreg_weight = sigreg_weight
        self.adaptive_tau = adaptive_tau

        # SIGReg internal buffers (same as the main SIGReg class)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    # ------------------------------------------------------------------
    def _sigreg_statistic(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute raw SIGReg GoF statistic (scalar). Lower = more Gaussian."""
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

    # ------------------------------------------------------------------
    def _adaptive_temperature(self, sigreg_stat: torch.Tensor) -> torch.Tensor:
        """Map SIGReg statistic to InfoNCE temperature via sigmoid gating.

        When sigreg_stat ≈ 0 (perfectly Gaussian, no collapse):
            tau → tau_base (sharp discrimination)
        When sigreg_stat is large (collapsing):
            tau → tau_max (soft, exploratory)
        """
        # Sigmoid gate: σ(stat - threshold) maps [0, ∞) → [0, 1)
        # Using stat itself (typically O(1)–O(10) range)
        gate = torch.sigmoid(sigreg_stat - 1.0)  # centered at stat=1
        tau = self.tau_base + (self.tau_max - self.tau_base) * gate
        return tau

    # ------------------------------------------------------------------
    def _pairwise_nce(self, z_a: torch.Tensor, z_b: torch.Tensor,
                      temperature: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE with adaptive temperature."""
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        B = z_a.shape[0]

        z = torch.cat([z_a, z_b], dim=0)  # (2B, D)
        sim = (z @ z.T) / temperature      # (2B, 2B)

        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(0, B, device=sim.device),
        ])
        return F.cross_entropy(sim, labels)

    # ------------------------------------------------------------------
    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (V, B, D) projected embeddings.

        Returns:
            Scalar loss combining adaptive InfoNCE + SIGReg.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        V, B, D = proj.shape

        # 1. Compute SIGReg statistic on the raw (un-normalized) projections
        sigreg_stat = self._sigreg_statistic(proj)

        # 2. Derive adaptive temperature
        if self.adaptive_tau:
            tau = self._adaptive_temperature(sigreg_stat.detach())
        else:
            tau = torch.tensor(self.tau_base, device=proj.device)

        # 3. InfoNCE with adaptive temperature
        if V < 2:
            nce_loss = torch.tensor(0.0, device=proj.device, requires_grad=True)
        else:
            nce_loss = torch.tensor(0.0, device=proj.device)
            n_pairs = 0
            for i in range(V):
                for j in range(i + 1, V):
                    nce_loss = nce_loss + self._pairwise_nce(proj[i], proj[j], tau)
                    n_pairs += 1
            nce_loss = nce_loss / max(n_pairs, 1)

        # 4. Adaptive SIGReg weight: scale up when distribution is unhealthy
        #    weight = base_weight * (1 + sigmoid(stat - 1))
        #    Range: [base_weight, 2*base_weight]
        adaptive_sig_weight = self.sigreg_weight * (
            1.0 + torch.sigmoid(sigreg_stat.detach() - 1.0)
        )
        sigreg_loss = sigreg_stat * adaptive_sig_weight

        loss = nce_loss + sigreg_loss
        return loss

    def forward_components(self, proj: torch.Tensor):
        """Return individual loss terms for logging."""
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        V, B, D = proj.shape

        sigreg_stat = self._sigreg_statistic(proj)

        if self.adaptive_tau:
            tau = self._adaptive_temperature(sigreg_stat.detach())
        else:
            tau = torch.tensor(self.tau_base, device=proj.device)

        if V < 2:
            nce_loss = torch.tensor(0.0, device=proj.device)
        else:
            nce_loss = torch.tensor(0.0, device=proj.device)
            n_pairs = 0
            for i in range(V):
                for j in range(i + 1, V):
                    nce_loss = nce_loss + self._pairwise_nce(proj[i], proj[j], tau)
                    n_pairs += 1
            nce_loss = nce_loss / max(n_pairs, 1)

        adaptive_sig_weight = self.sigreg_weight * (
            1.0 + torch.sigmoid(sigreg_stat.detach() - 1.0)
        )

        return {
            'infonce': nce_loss,
            'sigreg_stat': sigreg_stat,
            'adaptive_tau': tau,
            'adaptive_sigreg_weight': adaptive_sig_weight,
            'sigreg_loss': sigreg_stat * adaptive_sig_weight,
            'total': nce_loss + sigreg_stat * adaptive_sig_weight,
        }
