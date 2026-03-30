#!/usr/bin/env python3
"""
Neural Network Design Space Exploration for Cusp MLAT Prediction.
Explores multiple architectures and hyperparameters, reports best config.
"""
import json, glob, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================
# Data Loading
# ============================================================
def load_data():
    """Load OMNI-matched cusp crossings with physics + history features.
    Returns multi-target: eq_mlat, pole_mlat, eq_mlt, mean_mlt."""
    data_dir = os.environ.get('OMNI_DIR', 'output/omni_hist')
    filt_files = sorted(glob.glob(f'{data_dir}/cusp_crossings_*.json'))
    if not filt_files:
        filt_files = sorted(glob.glob('output/omni/cusp_crossings_*.json'))
    all_c = []
    for f in filt_files:
        with open(f) as fh:
            all_c.extend(json.load(fh))
    df = pd.DataFrame(all_c)

    # Targets
    df['abs_eq_mlat'] = df['eq_mlat'].abs()
    df['abs_pole_mlat'] = df['pole_mlat'].abs()
    df['hemi_code'] = (df['hemisphere'] == 'N').astype(float)
    df['doy'] = pd.to_datetime(df['time_start']).dt.dayofyear

    # Physics features
    df['B_T'] = np.sqrt(df['imf_by']**2 + df['imf_bz']**2)
    df['clock_angle'] = np.arctan2(df['imf_by'], df['imf_bz'])
    df['sin_clock_half'] = np.sin(df['clock_angle'] / 2)
    df['newell_cf'] = (df['sw_v'] ** (4/3)) * (df['B_T'] ** (2/3)) * (np.abs(df['sin_clock_half']) ** (8/3))
    df['kan_lee_ef'] = df['sw_v'] * df['B_T'] * (df['sin_clock_half'] ** 2)
    df['vBs'] = df['sw_v'] * np.where(df['imf_bz'] < 0, -df['imf_bz'], 0)
    df['by_hemi'] = df['imf_by'] * np.where(df['hemisphere'] == 'N', 1, -1)

    # Base features
    base_features = [
        'dipole_tilt', 'hemi_code', 'doy',
        'imf_bx', 'imf_by', 'imf_bz',
        'sw_v', 'sw_n', 'sw_pdyn',
        'B_T', 'clock_angle', 'sin_clock_half',
        'newell_cf', 'kan_lee_ef', 'vBs',
        'by_hemi',
    ]

    # History features (from add_omni_batch with time windows)
    hist_features = [c for c in df.columns if any(s in c for s in
        ['mean15','mean30','mean60','std15','std30','std60',
         'delta15','delta30','delta60','int60','_mean60'])]
    hist_features = [c for c in hist_features if c not in base_features]

    features = base_features + sorted(hist_features)
    targets = ['abs_eq_mlat', 'abs_pole_mlat', 'eq_mlt', 'mean_mlt']

    all_cols = features + targets + ['satellite']
    all_cols = [c for c in all_cols if c in df.columns]
    df_clean = df[all_cols].dropna()

    X = df_clean[[f for f in features if f in df_clean.columns]].values.astype(np.float32)
    y = df_clean[[t for t in targets if t in df_clean.columns]].values.astype(np.float32)
    sats = df_clean['satellite'].values
    used_features = [f for f in features if f in df_clean.columns]

    print(f"Features: {len(used_features)}, Targets: {y.shape[1]}, Samples: {len(X)}")
    return X, y, sats, used_features


# ============================================================
# Model Definitions
# ============================================================

class MLP(nn.Module):
    """Standard MLP with configurable layers."""
    def __init__(self, input_dim, hidden_dims, dropout=0.1, activation='relu', output_dim=4):
        super().__init__()
        layers = []
        prev = input_dim
        act_fn = nn.ReLU() if activation == 'relu' else nn.GELU() if activation == 'gelu' else nn.SiLU()
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), act_fn, nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, input_dim, hidden_dim, n_blocks, dropout=0.1, output_dim=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
            h = self.norm(h)
        return self.head(h)


class TabTransformer(nn.Module):
    """Lightweight transformer for tabular data."""
    def __init__(self, input_dim, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        # Each feature becomes a token
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4)
        )

    def forward(self, x):
        # x: (batch, n_features) -> tokens: (batch, n_features, d_model)
        tokens = torch.stack([
            emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeds)
        ], dim=1)
        h = self.transformer(tokens)
        # Mean pool
        h = h.mean(dim=1)
        return self.head(h)


# ============================================================
# Training Loop
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val,
                lr=1e-3, epochs=200, batch_size=256, patience=20, weight_decay=1e-4):
    """Train with early stopping, return best val MAE."""
    scaler_x = StandardScaler()
    X_tr = scaler_x.fit_transform(X_train)
    X_va = scaler_x.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = nn.L1Loss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_mae = nn.L1Loss()(val_pred, y_val_t).item()

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Restore best and compute final metrics
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    return mae, r2, epoch + 1, model


# ============================================================
# Design Space
# ============================================================

CONFIGS = [
    # --- MLP variants ---
    {'name': 'MLP-small', 'model_fn': lambda d: MLP(d, [64, 32]), 'lr': 1e-3, 'epochs': 300},
    {'name': 'MLP-medium', 'model_fn': lambda d: MLP(d, [128, 64, 32]), 'lr': 1e-3, 'epochs': 300},
    {'name': 'MLP-large', 'model_fn': lambda d: MLP(d, [256, 128, 64, 32]), 'lr': 5e-4, 'epochs': 400},
    {'name': 'MLP-wide', 'model_fn': lambda d: MLP(d, [512, 256]), 'lr': 5e-4, 'epochs': 300},
    {'name': 'MLP-deep', 'model_fn': lambda d: MLP(d, [128, 128, 128, 64, 32]), 'lr': 5e-4, 'epochs': 400},
    {'name': 'MLP-gelu', 'model_fn': lambda d: MLP(d, [128, 64, 32], activation='gelu'), 'lr': 1e-3, 'epochs': 300},
    {'name': 'MLP-silu', 'model_fn': lambda d: MLP(d, [128, 64, 32], activation='silu'), 'lr': 1e-3, 'epochs': 300},
    {'name': 'MLP-dropout0.2', 'model_fn': lambda d: MLP(d, [128, 64, 32], dropout=0.2), 'lr': 1e-3, 'epochs': 300},
    {'name': 'MLP-dropout0.3', 'model_fn': lambda d: MLP(d, [128, 64, 32], dropout=0.3), 'lr': 1e-3, 'epochs': 300},

    # --- Residual MLP variants ---
    {'name': 'ResMLP-2blk-64', 'model_fn': lambda d: ResidualMLP(d, 64, 2), 'lr': 1e-3, 'epochs': 300},
    {'name': 'ResMLP-3blk-64', 'model_fn': lambda d: ResidualMLP(d, 64, 3), 'lr': 1e-3, 'epochs': 300},
    {'name': 'ResMLP-4blk-64', 'model_fn': lambda d: ResidualMLP(d, 64, 4), 'lr': 5e-4, 'epochs': 400},
    {'name': 'ResMLP-2blk-128', 'model_fn': lambda d: ResidualMLP(d, 128, 2), 'lr': 5e-4, 'epochs': 300},
    {'name': 'ResMLP-3blk-128', 'model_fn': lambda d: ResidualMLP(d, 128, 3), 'lr': 5e-4, 'epochs': 400},
    {'name': 'ResMLP-4blk-128', 'model_fn': lambda d: ResidualMLP(d, 128, 4), 'lr': 3e-4, 'epochs': 400},

    # --- TabTransformer variants ---
    {'name': 'TabTF-d32-L1', 'model_fn': lambda d: TabTransformer(d, d_model=32, nhead=4, n_layers=1), 'lr': 1e-3, 'epochs': 200, 'batch_size': 512},
    {'name': 'TabTF-d64-L2', 'model_fn': lambda d: TabTransformer(d, d_model=64, nhead=4, n_layers=2), 'lr': 5e-4, 'epochs': 300, 'batch_size': 512},
    {'name': 'TabTF-d64-L3', 'model_fn': lambda d: TabTransformer(d, d_model=64, nhead=4, n_layers=3), 'lr': 3e-4, 'epochs': 400, 'batch_size': 512},
    {'name': 'TabTF-d128-L2', 'model_fn': lambda d: TabTransformer(d, d_model=128, nhead=8, n_layers=2), 'lr': 3e-4, 'epochs': 300, 'batch_size': 512},

    # --- Learning rate sweep on best architecture (MLP-medium) ---
    {'name': 'MLP-med-lr5e-4', 'model_fn': lambda d: MLP(d, [128, 64, 32]), 'lr': 5e-4, 'epochs': 400},
    {'name': 'MLP-med-lr2e-3', 'model_fn': lambda d: MLP(d, [128, 64, 32]), 'lr': 2e-3, 'epochs': 300},
    {'name': 'MLP-med-lr5e-3', 'model_fn': lambda d: MLP(d, [128, 64, 32]), 'lr': 5e-3, 'epochs': 200},
]


# ============================================================
# Main
# ============================================================

def main():
    TARGET_NAMES = ['eq_MLAT', 'pole_MLAT', 'eq_MLT', 'mean_MLT']
    TARGET_UNITS = ['°', '°', 'hr', 'hr']

    X, y, sats, feature_names = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    for i, (tn, tu) in enumerate(zip(TARGET_NAMES, TARGET_UNITS)):
        print(f"  {tn}: [{y[:,i].min():.1f}, {y[:,i].max():.1f}], mean={y[:,i].mean():.2f}{tu}")

    # Split: 70/15/15
    X_trainval, X_test, y_trainval, y_test, s_trainval, s_test = \
        train_test_split(X, y, sats, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_trainval, y_trainval, test_size=0.176, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    os.makedirs('dse_results', exist_ok=True)
    results = []

    for i, cfg in enumerate(CONFIGS):
        name = cfg['name']
        t0 = time.time()

        model = cfg['model_fn'](X.shape[1])
        n_params = sum(p.numel() for p in model.parameters())

        mae, r2, epochs_used, trained_model = train_model(
            model, X_train, y_train, X_val, y_val,
            lr=cfg['lr'],
            epochs=cfg['epochs'],
            batch_size=cfg.get('batch_size', 256),
            patience=cfg.get('patience', 25),
        )

        # Test set eval
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        trained_model.eval()
        with torch.no_grad():
            test_pred = trained_model(torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()

        elapsed = time.time() - t0

        # Per-target metrics
        row = {'name': name, 'n_params': n_params, 'epochs': epochs_used,
               'time_sec': elapsed, 'lr': cfg['lr'], 'val_mae_avg': mae, 'val_r2_avg': r2}
        per_target_str = []
        for ti, (tn, tu) in enumerate(zip(TARGET_NAMES, TARGET_UNITS)):
            t_mae = mean_absolute_error(y_test[:, ti], test_pred[:, ti])
            t_r2 = r2_score(y_test[:, ti], test_pred[:, ti])
            row[f'test_mae_{tn}'] = t_mae
            row[f'test_r2_{tn}'] = t_r2
            per_target_str.append(f"{tn}={t_mae:.3f}{tu}")
        row['test_mae_avg'] = np.mean([row[f'test_mae_{tn}'] for tn in TARGET_NAMES[:2]])  # avg MLAT MAE
        results.append(row)

        print(f"[{i+1}/{len(CONFIGS)}] {name:25s} | val={mae:.3f} | {' '.join(per_target_str)} | {n_params:,}p | {elapsed:.0f}s")

    # Save results
    df_res = pd.DataFrame(results).sort_values('test_mae_avg')
    df_res.to_csv('dse_results/dse_log.csv', index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RANKING (by avg MLAT test MAE)")
    print("=" * 80)
    for _, r in df_res.iterrows():
        parts = [f"{tn}={r.get(f'test_mae_{tn}', 0):.3f}" for tn in TARGET_NAMES]
        print(f"  {r['name']:25s} | MLAT_avg={r['test_mae_avg']:.3f}° | {' '.join(parts)} | {r['n_params']:,}p | {r['time_sec']:.0f}s")

    best = df_res.iloc[0]
    print(f"\n🏆 BEST: {best['name']} — avg MLAT MAE={best['test_mae_avg']:.3f}°")

    # GBR baseline (per target)
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    gbr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42))
    gbr.fit(X_tr_s, y_train)
    gbr_pred = gbr.predict(X_te_s)
    print(f"\n📊 GBR Baseline:")
    for ti, (tn, tu) in enumerate(zip(TARGET_NAMES, TARGET_UNITS)):
        g_mae = mean_absolute_error(y_test[:, ti], gbr_pred[:, ti])
        g_r2 = r2_score(y_test[:, ti], gbr_pred[:, ti])
        print(f"  {tn}: MAE={g_mae:.3f}{tu}, R²={g_r2:.4f}")


if __name__ == '__main__':
    main()
