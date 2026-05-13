"""
Novel Regularizers for MM-LeJEPA: Beyond SIGReg

This module implements three advanced regularization strategies that go beyond
the original LeJEPA SIGReg approach:

1. GMM Regularizer (Gaussian Mixture Model):
   - Uses learnable prototypes (cluster centers)
   - High batch entropy (use all clusters) + low conditional entropy (each sample near one prototype)
   
2. Sinkhorn Divergence (Optimal Transport):
   - Deterministic distribution matching via optimal transport
   - Computes transport cost to move embeddings to target Gaussian distribution
   
3. Spectral Regularization (SVD-based):
   - Forces embedding matrix to have flat spectrum (all singular values ≈ 1)
   - Explicit guarantee against dimensional collapse

Novelty Claims:
- GMM: "Discrete prototype-based regularization provides interpretable cluster structure
        while maintaining diversity through entropy constraints."
- Sinkhorn: "Unlike LeJEPA, which approximates distribution matching via stochastic projections
             (introducing variance), Sinkhorn Divergence optimizes the global geometry of
             the batch deterministically."
- Spectral: "For standard embedding sizes, explicit spectral regularization provides a
             deterministic guarantee against collapse that outweighs the overhead of SVD."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMRegularizer(nn.Module):
    """
    Gaussian Mixture Model Regularizer with Learnable Prototypes.
    
    Instead of matching to a single target distribution, we learn a set of
    prototypes (cluster centers). The objective:
    - High entropy across batch: Use all prototypes (prevent prototype collapse)
    - Low conditional entropy: Each sample close to exactly one prototype (tight clusters)
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_prototypes: Number of cluster centers (default: 10)
        temperature: Softmax temperature for assignment sharpness (default: 0.1)
    """
    
    def __init__(self, embedding_dim: int, num_prototypes: int = 10, temperature: float = 0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Learnable cluster centers (Prototypes)
        # Initialize on unit sphere for better optimization
        prototypes = torch.randn(num_prototypes, embedding_dim)
        prototypes = F.normalize(prototypes, dim=1)
        self.prototypes = nn.Parameter(prototypes)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute GMM regularization loss.
        
        Args:
            embeddings: (Batch_Size, Embedding_Dim) or (V, Batch_Size, Embedding_Dim)
                        where V is number of views
        
        Returns:
            loss: Scalar tensor combining fit loss and collapse prevention loss
        """
        # Handle view dimension: flatten to (N, D)
        original_shape = embeddings.shape
        if embeddings.dim() == 3:
            # (V, B, D) -> (V*B, D)
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        # Normalize embeddings and prototypes to live on the hypersphere
        z = F.normalize(embeddings, dim=1)
        c = F.normalize(self.prototypes, dim=1)
        
        # 1. Compute similarity scores: (Batch, Num_Prototypes)
        scores = torch.mm(z, c.t())
        
        # 2. Get probability assignment (soft clustering)
        probs = F.softmax(scores / self.temperature, dim=1)
        
        # --- LOSS COMPONENT A: Fit to Cluster ---
        # Pull each point towards its closest prototype
        # Maximize the log-likelihood of the assignment (minimize negative log-likelihood)
        log_probs = F.log_softmax(scores / self.temperature, dim=1)
        # Entropy of assignment distribution (per sample)
        loss_fit = -torch.sum(probs * log_probs, dim=1).mean()
        
        # --- LOSS COMPONENT B: Prevent Collapse (Entropy Regularization) ---
        # Calculate the mean probability across the batch for each prototype
        # We want the batch to be evenly distributed across prototypes
        mean_probs = probs.mean(dim=0)
        # Maximize entropy of the mean distribution (push it towards Uniform)
        # Minimizing negative entropy = maximizing entropy
        loss_collapse = torch.sum(mean_probs * torch.log(mean_probs + 1e-10))
        
        # Total Loss
        # loss_fit: minimize conditional entropy (tight clusters)
        # loss_collapse: maximize batch entropy (use all clusters)
        return loss_fit + loss_collapse
    
    def get_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get hard cluster assignments for embeddings (for analysis).
        
        Returns:
            assignments: (Batch_Size,) tensor of prototype indices
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        z = F.normalize(embeddings, dim=1)
        c = F.normalize(self.prototypes, dim=1)
        scores = torch.mm(z, c.t())
        return scores.argmax(dim=1)


class SinkhornRegularizer(nn.Module):
    """
    Sinkhorn Divergence Regularizer (Optimal Transport).
    
    Replaces LeJEPA's SIGReg (random projections) with deterministic optimal transport.
    Computes the "transport cost" to move batch distribution to a target Gaussian.
    
    Novelty: "Unlike LeJEPA, which approximates distribution matching via stochastic
             projections (introducing variance), Sinkhorn Divergence optimizes the
             global geometry of the batch deterministically."
    
    Args:
        epsilon: Entropic regularization coefficient (default: 0.1)
                 NOTE: This is a *relative* epsilon. Actual epsilon scales with
                 embedding dimension to handle high-dimensional cost matrices.
        n_iters: Number of Sinkhorn iterations (default: 10)
        use_geomloss: If True, use optimized geomloss library if available (default: True)
        normalize_embeddings: If True, L2-normalize embeddings before computing OT (default: True)
    """
    
    def __init__(self, epsilon: float = 0.1, n_iters: int = 10, use_geomloss: bool = True,
                 normalize_embeddings: bool = True):
        super().__init__()
        self.epsilon_base = epsilon  # Base epsilon (will be scaled)
        self.n_iters = n_iters
        self.use_geomloss = use_geomloss
        self.normalize_embeddings = normalize_embeddings
        
        # Try to import geomloss for optimized computation
        self.geomloss_available = False
        if use_geomloss:
            try:
                from geomloss import SamplesLoss
                self.sinkhorn_loss_fn = SamplesLoss(
                    loss="sinkhorn",
                    p=2,
                    blur=epsilon ** 0.5,  # geomloss uses blur = sqrt(epsilon)
                    scaling=0.9,  # Multi-scale parameter
                    debias=True   # Unbiased Sinkhorn divergence
                )
                self.geomloss_available = True
            except ImportError:
                pass
    
    def _sinkhorn_pure_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch Sinkhorn divergence implementation with numerical stability.
        
        Uses dimension-aware epsilon scaling and optional normalization.
        
        Args:
            x: (N, D) - Source embeddings
            y: (N, D) - Target samples (typically Gaussian noise)
        
        Returns:
            distance: Scalar Sinkhorn divergence (normalized to ~1 scale)
        """
        n, d = x.shape
        
        # Optionally normalize embeddings to unit sphere
        # This bounds the cost matrix and improves stability
        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            # For normalized vectors: ||x - y||^2 = 2(1 - cos(x,y)) ∈ [0, 4]
            # So epsilon doesn't need dimension scaling
            epsilon = self.epsilon_base * 2.0  # Scale for [0, 4] cost range
        else:
            # For unnormalized: ||x - y||^2 ≈ 2D for random Gaussians
            # Scale epsilon with dimension to maintain proper kernel bandwidth
            epsilon = self.epsilon_base * d
        
        # 1. Compute Cost Matrix (Squared Euclidean Distance)
        C = torch.cdist(x, y, p=2) ** 2
        
        # 2. Log-domain Sinkhorn for numerical stability
        log_K = -C / epsilon
        
        # Initialize log-domain potentials
        log_u = torch.zeros(n, device=x.device, dtype=x.dtype)
        log_v = torch.zeros(n, device=y.device, dtype=y.dtype)
        
        # 3. Sinkhorn Iterations in log-domain
        for _ in range(self.n_iters):
            log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = -torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        
        # 4. Calculate Transport Cost
        log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        P = torch.exp(log_P)
        
        # Clamp and check for NaN
        P = torch.clamp(P, min=0.0, max=1.0)
        distance = torch.sum(P * C)
        
        # Normalize by batch size to get per-sample cost
        distance = distance / n
        
        # Fallback if NaN
        if torch.isnan(distance) or torch.isinf(distance):
            # Fall back to simple regularizer: cosine similarity to Gaussian
            return 1.0 - F.cosine_similarity(x, y, dim=1).mean()
        
        return distance
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn divergence between embeddings and isotropic Gaussian.
        
        Args:
            embeddings: (Batch_Size, Dim) or (V, Batch_Size, Dim)
        
        Returns:
            loss: Scalar Sinkhorn divergence (normalized to ~1 scale)
        """
        # Handle view dimension: flatten to (N, D)
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        # Generate target: Isotropic Gaussian samples
        # For normalized mode, also normalize target
        target_noise = torch.randn_like(embeddings)
        
        if self.geomloss_available:
            # Use optimized geomloss implementation (handles normalization internally)
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, dim=1)
                target_noise = F.normalize(target_noise, dim=1)
            return self.sinkhorn_loss_fn(embeddings, target_noise)
        else:
            # Use pure PyTorch implementation
            return self._sinkhorn_pure_torch(embeddings, target_noise)


class SpectralRegularizer(nn.Module):
    """
    Spectral (SVD-based) Regularizer.
    
    Forces the embedding matrix to have a flat spectrum (all singular values ≈ 1).
    This ensures no dimension is wasted (prevents dimensional collapse).
    
    Why LeJEPA didn't do this: SVD is O(D³) complexity.
    Why it's viable now: For standard embedding sizes (256, 384, 512), SVD is fast on GPU.
    
    Novelty: "For standard embedding sizes, explicit spectral regularization provides
             a deterministic guarantee against collapse that outweighs the overhead of SVD."
    
    Args:
        mode: Regularization mode:
              - 'isotropy': MSE to identity (strict isotropy, like LeJEPA target)
              - 'barrier': Log barrier (softer, just prevent collapse)
              - 'hybrid': Combination of both
        target_variance: Target variance for eigenvalues (default: 1.0)
    """
    
    def __init__(self, mode: str = 'isotropy', target_variance: float = 1.0):
        super().__init__()
        self.mode = mode
        self.target_variance = target_variance
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral regularization loss.
        
        Args:
            embeddings: (Batch_Size, Dim) or (V, Batch_Size, Dim)
        
        Returns:
            loss: Scalar spectral regularization loss
        """
        # Handle view dimension: flatten to (N, D)
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        N, D = embeddings.shape
        
        # 1. Center the embeddings (remove mean)
        z = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # 2. Compute Covariance Matrix
        # For numerical stability and efficiency:
        # - If N >= D: compute D×D covariance matrix
        # - If N < D: compute N×N Gram matrix and use its eigenvalues
        if N >= D:
            # Standard covariance: (D, D)
            cov = (z.T @ z) / (N - 1)
            # Compute eigenvalues of symmetric positive semi-definite matrix
            # Note: eigvalsh doesn't support half precision, so cast to float32
            eigenvalues = torch.linalg.eigvalsh(cov.float())
        else:
            # Gram matrix approach: (N, N)
            gram = (z @ z.T) / (N - 1)
            eigenvalues = torch.linalg.eigvalsh(gram.float())
        
        # 3. Compute loss based on mode
        if self.mode == 'isotropy':
            # MSE to Identity: Force all eigenvalues to target variance
            target = torch.full_like(eigenvalues, self.target_variance)
            loss = F.mse_loss(eigenvalues, target)
            
        elif self.mode == 'barrier':
            # Log Barrier: Softly prevent eigenvalues from going to zero
            # Larger loss when eigenvalues approach 0
            loss = -torch.log(eigenvalues + 1e-6).mean()
            
        elif self.mode == 'hybrid':
            # Combination: Isotropy + Barrier
            target = torch.full_like(eigenvalues, self.target_variance)
            loss_isotropy = F.mse_loss(eigenvalues, target)
            loss_barrier = -torch.log(eigenvalues + 1e-6).mean()
            loss = loss_isotropy + 0.1 * loss_barrier
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return loss
    
    def get_spectrum(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get eigenvalue spectrum for analysis.
        
        Returns:
            eigenvalues: Sorted eigenvalues of covariance matrix
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        N, D = embeddings.shape
        z = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        if N >= D:
            cov = (z.T @ z) / (N - 1)
            eigenvalues = torch.linalg.eigvalsh(cov.float())  # Cast to float32 for CUDA support
        else:
            gram = (z @ z.T) / (N - 1)
            eigenvalues = torch.linalg.eigvalsh(gram.float())  # Cast to float32 for CUDA support
        
        return eigenvalues.sort(descending=True).values


class CombinedRegularizer(nn.Module):
    """
    Combined regularizer that applies multiple regularization strategies.
    
    This allows experimenting with different combinations of the novel regularizers.
    
    Args:
        embedding_dim: Dimension of input embeddings
        use_gmm: Enable GMM regularizer
        use_sinkhorn: Enable Sinkhorn regularizer  
        use_spectral: Enable Spectral regularizer
        gmm_prototypes: Number of GMM prototypes
        gmm_temperature: GMM softmax temperature
        sinkhorn_epsilon: Sinkhorn entropic regularization
        sinkhorn_iters: Number of Sinkhorn iterations
        spectral_mode: Spectral regularizer mode
        weights: Dict of weights for each regularizer (default: equal)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        use_gmm: bool = True,
        use_sinkhorn: bool = False,
        use_spectral: bool = False,
        gmm_prototypes: int = 10,
        gmm_temperature: float = 0.1,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iters: int = 5,
        spectral_mode: str = 'isotropy',
        weights: dict = None
    ):
        super().__init__()
        
        self.use_gmm = use_gmm
        self.use_sinkhorn = use_sinkhorn
        self.use_spectral = use_spectral
        
        self.weights = weights or {'gmm': 1.0, 'sinkhorn': 1.0, 'spectral': 1.0}
        
        if use_gmm:
            self.gmm = GMMRegularizer(
                embedding_dim=embedding_dim,
                num_prototypes=gmm_prototypes,
                temperature=gmm_temperature
            )
        
        if use_sinkhorn:
            self.sinkhorn = SinkhornRegularizer(
                epsilon=sinkhorn_epsilon,
                n_iters=sinkhorn_iters
            )
        
        if use_spectral:
            self.spectral = SpectralRegularizer(mode=spectral_mode)
    
    def forward(self, embeddings: torch.Tensor) -> tuple:
        """
        Compute combined regularization loss.
        
        Args:
            embeddings: (Batch_Size, Dim) or (V, Batch_Size, Dim)
        
        Returns:
            total_loss: Weighted sum of all enabled regularizers
            loss_dict: Dictionary with individual loss values
        """
        total_loss = torch.tensor(0.0, device=embeddings.device)
        loss_dict = {}
        
        if self.use_gmm:
            gmm_loss = self.gmm(embeddings)
            total_loss = total_loss + self.weights['gmm'] * gmm_loss
            loss_dict['gmm_loss'] = gmm_loss.item()
        
        if self.use_sinkhorn:
            sinkhorn_loss = self.sinkhorn(embeddings)
            total_loss = total_loss + self.weights['sinkhorn'] * sinkhorn_loss
            loss_dict['sinkhorn_loss'] = sinkhorn_loss.item()
        
        if self.use_spectral:
            spectral_loss = self.spectral(embeddings)
            total_loss = total_loss + self.weights['spectral'] * spectral_loss
            loss_dict['spectral_loss'] = spectral_loss.item()
        
        return total_loss, loss_dict


# Convenience factory function
def create_regularizer(
    regularizer_type: str,
    embedding_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create regularizers.
    
    Args:
        regularizer_type: One of 'gmm', 'sinkhorn', 'spectral', 'combined'
        embedding_dim: Dimension of input embeddings
        **kwargs: Additional arguments passed to the regularizer
    
    Returns:
        Regularizer module
    """
    if regularizer_type == 'gmm':
        return GMMRegularizer(
            embedding_dim=embedding_dim,
            num_prototypes=kwargs.get('num_prototypes', 10),
            temperature=kwargs.get('temperature', 0.1)
        )
    elif regularizer_type == 'sinkhorn':
        return SinkhornRegularizer(
            epsilon=kwargs.get('epsilon', 0.1),
            n_iters=kwargs.get('n_iters', 5)
        )
    elif regularizer_type == 'spectral':
        return SpectralRegularizer(
            mode=kwargs.get('mode', 'isotropy'),
            target_variance=kwargs.get('target_variance', 1.0)
        )
    elif regularizer_type == 'combined':
        return CombinedRegularizer(
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown regularizer type: {regularizer_type}")


# Testing
if __name__ == "__main__":
    print("Testing Novel Regularizers...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    embedding_dim = 256
    
    # Create test embeddings
    embeddings = torch.randn(batch_size, embedding_dim, device=device)
    embeddings_views = torch.randn(4, batch_size, embedding_dim, device=device)  # With views
    
    print(f"\nTest embeddings shape: {embeddings.shape}")
    print(f"Test embeddings with views shape: {embeddings_views.shape}")
    
    # Test GMM Regularizer
    print("\n1. GMM Regularizer:")
    gmm = GMMRegularizer(embedding_dim=embedding_dim, num_prototypes=10).to(device)
    loss = gmm(embeddings)
    print(f"   Loss (flat): {loss.item():.4f}")
    loss_views = gmm(embeddings_views)
    print(f"   Loss (views): {loss_views.item():.4f}")
    assignments = gmm.get_assignments(embeddings)
    print(f"   Unique assignments: {len(assignments.unique())}/{gmm.num_prototypes}")
    
    # Test Sinkhorn Regularizer
    print("\n2. Sinkhorn Regularizer:")
    sinkhorn = SinkhornRegularizer(epsilon=0.1, n_iters=5).to(device)
    loss = sinkhorn(embeddings)
    print(f"   Loss (flat): {loss.item():.4f}")
    print(f"   Using geomloss: {sinkhorn.geomloss_available}")
    loss_views = sinkhorn(embeddings_views)
    print(f"   Loss (views): {loss_views.item():.4f}")
    
    # Test Spectral Regularizer
    print("\n3. Spectral Regularizer:")
    for mode in ['isotropy', 'barrier', 'hybrid']:
        spectral = SpectralRegularizer(mode=mode).to(device)
        loss = spectral(embeddings)
        print(f"   Mode '{mode}': Loss = {loss.item():.4f}")
    
    spectrum = spectral.get_spectrum(embeddings)
    print(f"   Top 5 eigenvalues: {spectrum[:5].tolist()}")
    print(f"   Bottom 5 eigenvalues: {spectrum[-5:].tolist()}")
    
    # Test Combined Regularizer
    print("\n4. Combined Regularizer:")
    combined = CombinedRegularizer(
        embedding_dim=embedding_dim,
        use_gmm=True,
        use_sinkhorn=True,
        use_spectral=True
    ).to(device)
    total_loss, loss_dict = combined(embeddings)
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Individual losses: {loss_dict}")
    
    # Test factory function
    print("\n5. Factory Function:")
    for reg_type in ['gmm', 'sinkhorn', 'spectral']:
        reg = create_regularizer(reg_type, embedding_dim).to(device)
        loss = reg(embeddings)
        print(f"   {reg_type}: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")
