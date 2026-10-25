from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fifa_value.features.sequences import PlayerSequenceDataset


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float) -> torch.Tensor:
    """Pinball (quantile) loss for a given quantile alpha."""
    err = y_true - y_pred
    return torch.max(alpha * err, (alpha - 1.0) * err).mean()


class SequencePlayerModel(nn.Module):
    """
    Multi-task sequence neural network:
    - Encodes L-season player history using a GRU
    - Concatenates with current season static context
    - Produces joint outputs for Target A, Target B, and Quantiles (q05, q50, q95)
    """

    def __init__(
        self,
        d_seq: int,
        d_static: int,
        d_model: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(d_seq, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.static_proj = nn.Sequential(
            nn.Linear(d_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_dim = hidden_dim + 32
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Output heads
        self.head_a = nn.Linear(64, 1)      # Level target
        self.head_b = nn.Linear(64, 1)      # Growth target
        self.head_q05 = nn.Linear(64, 1)    # 5th percentile
        self.head_q50 = nn.Linear(64, 1)    # 50th percentile
        self.head_q95 = nn.Linear(64, 1)    # 95th percentile

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor,
        x_static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Zero out unobserved time steps
        masked_x = x_seq * mask.unsqueeze(-1)
        h_seq = torch.relu(self.seq_proj(masked_x))
        out, _ = self.gru(h_seq)

        # Take last time step (season t)
        last_h = out[:, -1, :]

        h_stat = self.static_proj(x_static)
        combined = torch.cat([last_h, h_stat], dim=-1)
        rep = self.backbone(combined)

        return {
            "pred_a": self.head_a(rep).squeeze(-1),
            "pred_b": self.head_b(rep).squeeze(-1),
            "q05": self.head_q05(rep).squeeze(-1),
            "q50": self.head_q50(rep).squeeze(-1),
            "q95": self.head_q95(rep).squeeze(-1),
        }


class SequenceNNBranch:
    """Training, inference, and seed-averaging wrapper for the sequence neural network."""

    def __init__(
        self,
        history_length: int = 4,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str | None = None,
        seeds: Sequence[int] = (42, 123),
    ):
        self.history_length = history_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seeds = list(seeds)
        self.models: list[SequencePlayerModel] = []

    def fit(
        self,
        train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "SequenceNNBranch":
        x_seq_tr, mask_tr, x_stat_tr, y_a_tr, y_b_tr = train_data
        train_ds = PlayerSequenceDataset(x_seq_tr, mask_tr, x_stat_tr, y_a_tr, y_b_tr)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            x_seq_val, mask_val, x_stat_val, y_a_val, y_b_val = val_data
            val_ds = PlayerSequenceDataset(x_seq_val, mask_val, x_stat_val, y_a_val, y_b_val)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        d_seq = x_seq_tr.shape[2]
        d_static = x_stat_tr.shape[1]

        self.models = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SequencePlayerModel(d_seq=d_seq, d_static=d_static).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            best_val_loss = float("inf")
            best_state = None

            for epoch in range(self.epochs):
                model.train()
                for batch in train_loader:
                    bx_seq = batch["x_seq"].to(self.device)
                    bmask = batch["mask"].to(self.device)
                    bx_stat = batch["x_static"].to(self.device)
                    by_a = batch["y_a"].to(self.device)
                    by_b = batch["y_b"].to(self.device)

                    optimizer.zero_grad()
                    out = model(bx_seq, bmask, bx_stat)

                    loss_a = nn.functional.mse_loss(out["pred_a"], by_a)
                    loss_b = nn.functional.mse_loss(out["pred_b"], by_b)
                    loss_q05 = pinball_loss(by_a, out["q05"], 0.05)
                    loss_q50 = pinball_loss(by_a, out["q50"], 0.50)
                    loss_q95 = pinball_loss(by_a, out["q95"], 0.95)

                    total_loss = loss_a + loss_b + 0.5 * (loss_q05 + loss_q50 + loss_q95)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                # Validation step
                if val_loader is not None:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in val_loader:
                            bx_seq = batch["x_seq"].to(self.device)
                            bmask = batch["mask"].to(self.device)
                            bx_stat = batch["x_static"].to(self.device)
                            by_a = batch["y_a"].to(self.device)
                            out = model(bx_seq, bmask, bx_stat)
                            v_loss = nn.functional.mse_loss(out["pred_a"], by_a)
                            val_losses.append(v_loss.item())
                    avg_v_loss = np.mean(val_losses)
                    if avg_v_loss < best_val_loss:
                        best_val_loss = avg_v_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)
            self.models.append(model)

        return self

    def predict(
        self,
        data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    ) -> dict[str, np.ndarray]:
        """Averages predictions across all trained seed models."""
        x_seq, mask, x_stat = data[0], data[1], data[2]
        ds = PlayerSequenceDataset(x_seq, mask, x_stat)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds_a, preds_b, q05s, q50s, q95s = [], [], [], [], []

        for model in self.models:
            model.eval()
            m_a, m_b, m_q05, m_q50, m_q95 = [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    bx_seq = batch["x_seq"].to(self.device)
                    bmask = batch["mask"].to(self.device)
                    bx_stat = batch["x_static"].to(self.device)
                    out = model(bx_seq, bmask, bx_stat)

                    m_a.append(out["pred_a"].cpu().numpy())
                    m_b.append(out["pred_b"].cpu().numpy())
                    m_q05.append(out["q05"].cpu().numpy())
                    m_q50.append(out["q50"].cpu().numpy())
                    m_q95.append(out["q95"].cpu().numpy())

            preds_a.append(np.concatenate(m_a))
            preds_b.append(np.concatenate(m_b))
            q05s.append(np.concatenate(m_q05))
            q50s.append(np.concatenate(m_q50))
            q95s.append(np.concatenate(m_q95))

        return {
            "pred_a": np.mean(preds_a, axis=0),
            "pred_b": np.mean(preds_b, axis=0),
            "q05": np.mean(q05s, axis=0),
            "q50": np.mean(q50s, axis=0),
            "q95": np.mean(q95s, axis=0),
        }
