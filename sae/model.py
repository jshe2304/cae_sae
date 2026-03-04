import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, n_input, n_latent, k, aux_k=512):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.k = k
        self.aux_k = aux_k

        # Shared pre-encoder bias (subtracted before encoding, added after decoding)
        self.b = nn.Parameter(torch.zeros(n_input))

        # Encoder: linear map + bias
        self.encoder = nn.Linear(n_input, n_latent, bias=True)

        # Decoder: linear map, no bias (bias handled by self.b)
        self.decoder = nn.Linear(n_latent, n_input, bias=False)

        # Dead feature tracking (not parameters, but persistent state)
        self.register_buffer("miss_counts", torch.zeros(n_latent, dtype=torch.long))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)  # (n_input, n_latent)
        # Initialize encoder as transpose of decoder
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        self._normalize_decoder()

    @staticmethod
    def topk_activation(x, k):
        topk_vals, topk_idx = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def forward(self, x):
        # Encode
        x_centered = x - self.b
        alpha_pre = self.encoder(x_centered)

        # TopK activation
        alpha = self.topk_activation(alpha_pre, self.k)

        # Decode
        x_hat = self.decoder(alpha) + self.b

        # Track which features fired
        fired_mask = alpha != 0  # (batch, n_latent)

        # Dead feature mask
        dead_mask = self.miss_counts >= self._effective_threshold()

        # Auxiliary reconstruction on dead features (for dead feature loss)
        aux_x_hat = None
        if dead_mask.any():
            # Only use dead features to reconstruct the residual
            alpha_pre_dead = alpha_pre.clone()
            alpha_pre_dead[:, ~dead_mask] = float("-inf")
            alpha_aux = self.topk_activation(alpha_pre_dead, min(self.aux_k, dead_mask.sum().item()))
            aux_x_hat = self.decoder(alpha_aux) + self.b

        return x_hat, alpha, fired_mask, aux_x_hat

    def _effective_threshold(self):
        return self.miss_counts.new_tensor(1).item()  # minimum 1 to avoid issues at step 0

    def update_dead_mask(self, fired_mask):
        """Update miss counts: increment for inactive features, reset for active."""
        any_fired = fired_mask.any(dim=0)  # (n_latent,)
        self.miss_counts[any_fired] = 0
        self.miss_counts[~any_fired] += fired_mask.shape[0]  # increment by batch size
        self.step += 1

    def project_decoder_grads(self):
        """Remove gradient component parallel to decoder columns (maintain unit norm direction)."""
        if self.decoder.weight.grad is None:
            return
        W = self.decoder.weight.data  # (n_input, n_latent)
        grad = self.decoder.weight.grad
        # For each column j, remove the component of grad along W[:,j]
        # dot product per column: sum over n_input dimension
        dots = (grad * W).sum(dim=0, keepdim=True)  # (1, n_latent)
        grad.sub_(W * dots)

    def _normalize_decoder(self):
        """Enforce unit L2 norm on each decoder column."""
        with torch.no_grad():
            W = self.decoder.weight.data  # (n_input, n_latent)
            norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W.div_(norms)
