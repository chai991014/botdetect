import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from torch_geometric.nn import HypergraphConv
import copy


# =============================================================================
# 1. SETUP & DATA CONSTRUCTION
# =============================================================================
class DualLogger(object):
    """
    Redirects stdout to both the terminal and a log file.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(output_dir, file_name):
    """
    Sets up the logger to write to a file in output_dir.
    Returns the full path to the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"log_{file_name}.txt")
    sys.stdout = DualLogger(log_file_path)
    return log_file_path


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logger("./result", f"BotSTHCL_{timestamp}")

DATA_PATH = "./datasets"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_df = pd.read_csv(f"{DATA_PATH}/train_features.csv")
test_df = pd.read_csv(f"{DATA_PATH}/test_features.csv")
raw_bids = pd.read_csv(f"{DATA_PATH}/bids.csv")

# Global Indexing
all_bidders = pd.concat([train_df['bidder_id'], test_df['bidder_id']]).unique()
bidder_to_idx = {bidder: i for i, bidder in enumerate(all_bidders)}
num_nodes = len(all_bidders)

# Feature Preparation
full_features = pd.concat([train_df, test_df]).set_index('bidder_id').reindex(all_bidders)
X_raw = full_features.drop(columns=['payment_account', 'address', 'outcome'], errors='ignore').fillna(0)

# Extract Temporal Indices
temporal_cols = []
for i in range(1, 6):
    temporal_cols.extend([f'burst_max_strip{i}', f'burst_mean_strip{i}', f'burst_ratio_strip{i}'])
temporal_indices = [X_raw.columns.get_loc(col) for col in temporal_cols if col in X_raw.columns]
print(f"Detected {len(temporal_indices)} temporal features for 5 time strips.")

scaler = StandardScaler()
x = torch.from_numpy(scaler.fit_transform(X_raw)).float().to(device)

# Hyperedge Construction (Auction + IP)
unique_auctions = raw_bids['auction'].unique()
auction_to_idx = {auc: i for i, auc in enumerate(unique_auctions)}
unique_ips = raw_bids['ip'].unique()
ip_to_idx = {ip: i + len(unique_auctions) for i, ip in enumerate(unique_ips)}
unique_devices = raw_bids['device'].unique()
dev_to_idx = {dev: i + len(unique_auctions) + len(unique_ips) for i, dev in enumerate(unique_devices)}

valid_bids = raw_bids[raw_bids['bidder_id'].isin(bidder_to_idx)].copy()
nodes = np.concatenate([valid_bids['bidder_id'].map(bidder_to_idx).values] * 3)  # x3 now
hyperedges = np.concatenate([
    valid_bids['auction'].map(auction_to_idx).values,
    valid_bids['ip'].map(ip_to_idx).values,
    valid_bids['device'].map(dev_to_idx).values  # Added device edges
])
edge_index = torch.from_numpy(np.stack([nodes, hyperedges])).long().to(device)

edge_counts = torch.bincount(edge_index[1])
hyperedge_weight = 1.0 / torch.log(edge_counts.float() + 1.5)
hyperedge_weight = hyperedge_weight / hyperedge_weight.mean()
hyperedge_weight = hyperedge_weight.to(device)

print(f"Hyperedge weights calculated for {len(hyperedge_weight)} unique auctions/IPs/Devices.")
print(f"Hyperedge weights normalized. Mean: {hyperedge_weight.mean():.4f}")

# Labels
labels = np.full(num_nodes, -1)
for _, row in train_df.iterrows():
    labels[bidder_to_idx[row['bidder_id']]] = int(row['outcome'])
y = torch.tensor(labels, dtype=torch.long).to(device)


# =============================================================================
# 2. CONTRASTIVE LOSS & ST-HGNN NETWORK
# =============================================================================
def info_nce_loss(z1, z2, temperature=0.5, sample_size=1024):
    """
    Computes InfoNCE loss with in-batch negatives for Contrastive Learning.
    Limits sample size to prevent OOM errors on large hypergraphs.
    """
    num_nodes = z1.size(0)
    if num_nodes > sample_size:
        idx = torch.randperm(num_nodes, device=z1.device)[:sample_size]
        z1 = z1[idx]
        z2 = z2[idx]

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Cosine similarity matrix
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(logits.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)


class STHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_indices=None, seq_len=5):
        super().__init__()
        self.temporal_indices = temporal_indices if temporal_indices is not None else []
        self.static_indices = [i for i in range(in_channels) if i not in self.temporal_indices]
        self.seq_len = seq_len
        self.temp_dim = len(self.temporal_indices) // seq_len if len(self.temporal_indices) > 0 else 0

        # Two-layer spatial branch for deeper community detection
        self.spatial_conv1 = HypergraphConv(in_channels, hidden_channels)
        self.spatial_conv2 = HypergraphConv(hidden_channels, hidden_channels)

        # Stability layers
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        # Projector head for Contrastive Learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Temporal branch (Transformer)
        if self.temp_dim > 0:
            self.temp_proj = nn.Linear(self.temp_dim, hidden_channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=4,
                dim_feedforward=hidden_channels * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Behavioral branch
        static_dim = len(self.static_indices) if len(self.static_indices) > 0 else in_channels
        self.behavioral_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        # Squeeze-and-Excitation Gating
        self.gate_fc1 = nn.Linear(hidden_channels * 2, hidden_channels // 2)
        self.gate_fc2 = nn.Linear(hidden_channels // 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # Separated spatial encoding logic for reuse in CL
    def encode_spatial(self, x, edge_index, edge_weight):
        z1 = F.leaky_relu(self.ln1(self.spatial_conv1(x, edge_index, hyperedge_weight=edge_weight)), 0.2)
        z2 = F.leaky_relu(self.ln2(self.spatial_conv2(z1, edge_index, hyperedge_weight=edge_weight)), 0.2)
        return z1 + z2

    def forward(self, x, edge_index, edge_weight, return_spatial=False):
        # View 1: Structural (Spatial)
        # Hyperedge Dropout (10%) during main forward pass for structural robustness
        if self.training:
            num_edges = edge_index.size(1)
            mask = torch.bernoulli(torch.full((num_edges,), 0.9)).to(x.device).bool()
            curr_edge_index = edge_index[:, mask]
        else:
            curr_edge_index = edge_index

        z_spatial = self.encode_spatial(x, curr_edge_index, edge_weight)

        # View 2: Behavioral (Static Features)
        x_static = x[:, self.static_indices]
        z_behavioral = self.behavioral_mlp(x_static)

        # View 3: Temporal Sequence Modeling
        if self.temp_dim > 0:
            x_temp = x[:, self.temporal_indices].view(-1, self.seq_len, self.temp_dim)
            x_temp_emb = self.temp_proj(x_temp)
            z_temp_seq = self.transformer(x_temp_emb)
            z_temporal = z_temp_seq.mean(dim=1)  # Pool across time

            # Combine individual behavioral traits with temporal evolution
            z_bt = z_behavioral + z_temporal
        else:
            z_bt = z_behavioral

        # Dynamic SE-Gate logic
        combined = torch.cat([z_spatial, z_bt], dim=1)
        se = F.relu(self.gate_fc1(combined))
        g = torch.sigmoid(self.gate_fc2(se))
        z_fused = g * z_spatial + (1 - g) * z_bt

        logits = self.classifier(z_fused)

        # Allow returning spatial embeddings for CL reuse
        if return_spatial:
            return logits, z_spatial
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=25, gamma=2, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents gradient explosion
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# =============================================================================
# 3. K-FOLD TRAINING ENGINE
# =============================================================================
def train_kfold(k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labeled_indices = np.where(labels != -1)[0]
    labeled_y = labels[labeled_indices]

    fold_aucs = []
    fold_aps = []
    oof_probs = np.zeros(len(labeled_indices))
    test_probs_acc = np.zeros(num_nodes)

    criterion = FocalLoss(alpha=25, gamma=2)

    print(f"Starting {k}-Fold Cross-Validation with Checkpointing...")

    EARLY_STOPPING_PATIENCE = 25

    # Contrastive learning hyperparams
    tau = 0.5
    lambda_cl = 0.1

    for fold, (t_idx, v_idx) in enumerate(skf.split(labeled_indices, labeled_y)):
        model = STHGNN(x.shape[1], 64, temporal_indices).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        f_train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        f_val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        f_train_mask[labeled_indices[t_idx]] = True
        f_val_mask[labeled_indices[v_idx]] = True

        best_auc = 0
        patience_counter = 0
        best_model_state = None

        print(f"\n--- FOLD {fold + 1} ---")

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()

            if epoch % 5 == 0:
                # 1. Main Pass (Reuse spatial embeddings)
                out, z_main = model(x, edge_index, hyperedge_weight, return_spatial=True)
                out = out.squeeze()
                cls_loss = criterion(out[f_train_mask], y[f_train_mask].float())

                # 2. Single Combined Augmentation (Mask attributes + Drop nodes)
                drop_mask = torch.rand(x.shape, device=device) < 0.1
                x_aug = x.clone()
                x_aug[drop_mask] = 0

                edge_mask = torch.rand(edge_index.size(1), device=device) > 0.1
                edge_index_aug = edge_index[:, edge_mask]

                # 3. Compute ONLY ONE extra spatial encoding
                z_aug = model.encode_spatial(x_aug, edge_index_aug, hyperedge_weight)

                # 4. Project and compute InfoNCE
                z1 = model.projector(z_main)
                z2 = model.projector(z_aug)
                cl_loss = info_nce_loss(z1, z2, temperature=tau, sample_size=512)

                # Combine loss (lowered lambda to 0.05 so CL doesn't overpower Cls loss)
                loss = cls_loss + (0.05 * cl_loss)
            else:
                # Fast standard forward pass
                out = model(x, edge_index, hyperedge_weight).squeeze()
                cls_loss = criterion(out[f_train_mask], y[f_train_mask].float())
                loss = cls_loss
                cl_loss = torch.tensor(0.0)  # Dummy for logging

            loss.backward()
            optimizer.step()

            # Validation for checkpointing
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index, hyperedge_weight).squeeze()
                v_probs_epoch = torch.sigmoid(val_out[f_val_mask]).cpu().numpy()
                v_labels_epoch = y[f_val_mask].cpu().numpy()
                val_auc = roc_auc_score(v_labels_epoch, v_probs_epoch)

                # Checkpoint the best version of the model
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                scheduler.step()

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, CL: {cl_loss.item():.4f}) | Val AUC: {val_auc:.4f} (Best: {best_auc:.4f})")

        # Reload the BEST model state before finalizing the fold
        model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            full_logits = model(x, edge_index, hyperedge_weight).squeeze()
            full_probs = torch.sigmoid(full_logits).cpu().numpy()

            v_probs = full_probs[labeled_indices[v_idx]]
            v_labels = labeled_y[v_idx]

            oof_probs[v_idx] = v_probs
            fold_aucs.append(roc_auc_score(v_labels, v_probs))
            fold_aps.append(average_precision_score(v_labels, v_probs))
            test_probs_acc += full_probs / k

        print(f"Fold {fold + 1} Finished. Best Fold AUC: {fold_aucs[-1]:.4f}")

    # =============================================================================
    # 4. FINAL SUMMARY
    # =============================================================================
    oof_auc = roc_auc_score(labeled_y, oof_probs)
    oof_ap = average_precision_score(labeled_y, oof_probs)

    # Optimized Threshold on OOF Predictions
    prec, rec, thresholds = precision_recall_curve(labeled_y, oof_probs)
    f1_scores = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (BotSTHCL)")
    print("=" * 60)
    print(f"  Model                    : BotSTHCL (Spatio-Temporal Hypergraph CL)")
    print(f"  CV Folds                 : {k}")
    print(f"  CV AUC (mean)            : {np.mean(fold_aucs):.4f}")
    print(f"  CV AUC (std)             : {np.std(fold_aucs):.4f}")
    print(f"  CV AP  (mean)            : {np.mean(fold_aps):.4f}")
    print(f"  OOF AUC                  : {oof_auc:.4f}")
    print(f"  OOF AP                   : {oof_ap:.4f}")
    print(f"  Best F1                  : {best_f1:.4f}")
    print(f"  Best Threshold           : {best_thresh:.4f}")
    print("=" * 60 + "\n")

    print(classification_report(labeled_y, (oof_probs > best_thresh).astype(int), target_names=['Human', 'Robot']))

    return test_probs_acc


# Execute and Save
final_test_probs = train_kfold(k=5)

test_bidders = test_df['bidder_id'].values
test_preds = [final_test_probs[bidder_to_idx[b_id]] for b_id in test_bidders]
pd.DataFrame({'bidder_id': test_bidders, 'prediction': test_preds}).to_csv("./result/botsthcl_final_oof_results.csv",
                                                                           index=False)

print("\nFinal submission generated: botsthcl_final_oof_results.csv")
