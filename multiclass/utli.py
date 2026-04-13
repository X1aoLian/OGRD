import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, use_bn=True):
        super().__init__()
        from torch.nn import ModuleList, Linear, BatchNorm1d

        self.num_layers = num_layers
        self.use_bn = use_bn
        self.linears = ModuleList()
        self.bns = ModuleList() if use_bn else None

        self.linears.append(Linear(input_channels, hidden_channels))
        if use_bn:
            self.bns.append(BatchNorm1d(hidden_channels))

        for layer in range(num_layers - 2):
            self.linears.append(Linear(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.linears.append(Linear(hidden_channels, output_channels))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)

        if self.num_layers > 1:
            x = self.linears[-1](x)
        return x


class GINSWDSingleModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            initial_atoms: torch.Tensor,
            n_projections: int = 100,
            seed: int = 0,
            mlp_layers: int = 3,
            use_bn: bool = True,
            train_eps: bool = False,
            swd_method: str = "mmd",
            gamma=None
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            mlp = MLP(inc, hidden_channels, hidden_channels, mlp_layers, use_bn)
            self.convs.append(GINConv(mlp, train_eps=train_eps))

        self.atoms = nn.Parameter(initial_atoms)

        C, K, D = initial_atoms.shape
        if gamma is None:
            self.gamma = 1.0 / D
        else:
            self.gamma = gamma

    def forward(self, data_or_x, batch=None, edge_index=None):
        if batch is None:
            data = data_or_x
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = conv(x, edge_index)
        else:
            x = data_or_x
        return self._mmd_distances(x, batch)

    def extract_node_features(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

    def _mmd_distances(self, x_mat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        device = x_mat.device
        C, K, D = self.atoms.shape
        bs = int(batch.max()) + 1

        mmd_distances = torch.zeros(bs, C, device=device)
        for i in range(bs):
            Xi = x_mat[batch == i]
            for j in range(C):
                Aj = self.atoms[j]
                mmd_distances[i, j] = self._mmd(Xi, Aj)
        return mmd_distances

    def _mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(2)
        k_xx = torch.exp(-gamma * xx).mean()
        yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        k_yy = torch.exp(-gamma * yy).mean()
        xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        k_xy = torch.exp(-gamma * xy).mean()
        return k_xx + k_yy - 2 * k_xy

    def atom_distances(self) -> torch.Tensor:
        C, K, D = self.atoms.shape
        device = self.atoms.device
        distances = torch.zeros(C, C, device=device)

        for i in range(C):
            for j in range(C):
                if i != j:
                    distances[i, j] = self._mmd(self.atoms[i], self.atoms[j])

        return distances


def init_atoms_from_dataset(
        dataset,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        num_atom_supp: int,
        device: torch.device
) -> torch.Tensor:
    convs = torch.nn.ModuleList()
    for i in range(num_layers):
        inc = in_channels if i == 0 else hidden_channels
        mlp = Sequential(
            Linear(inc, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        convs.append(GINConv(mlp))
    convs = convs.to(device)

    graph_embs, node_counts, labels = [], [], []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            x, edge_index = data.x, data.edge_index
            for conv in convs:
                x = conv(x, edge_index)
            graph_embs.append(x.cpu())
            node_counts.append(x.size(0))
            labels.append(int(data.y.item()))

    atoms = []
    for c in range(num_classes):
        idxs = [i for i, lbl in enumerate(labels) if lbl == c]
        sel = None
        for i in idxs:
            if node_counts[i] == num_atom_supp:
                sel = graph_embs[i]
                break
        if sel is not None:
            atoms.append(sel)
        else:
            all_nodes = torch.cat([graph_embs[i] for i in idxs], dim=0)
            km = KMeans(n_clusters=num_atom_supp, random_state=0).fit(all_nodes.numpy())
            centers = torch.from_numpy(km.cluster_centers_)
            atoms.append(centers)

    return torch.stack(atoms, dim=0)


def sample_anti_atoms_negatives(model, num_neg):
    atoms = model.atoms
    C, K, D = atoms.shape
    device = atoms.device

    atoms_center = atoms.mean(dim=(0, 1))

    neg_features_list = []
    neg_batch_list = []

    for i in range(num_neg):
        random_direction = torch.randn(D, device=device)
        random_direction = random_direction / random_direction.norm()

        distance = torch.rand(1, device=device) * 5 + 2
        neg_center = atoms_center + random_direction * distance

        neg_feat = neg_center.unsqueeze(0).repeat(10, 1)
        neg_feat += torch.randn_like(neg_feat) * 0.5

        neg_features_list.append(neg_feat)
        neg_batch_list.append(torch.full((10,), i, dtype=torch.long, device=device))

    return torch.cat(neg_features_list), torch.cat(neg_batch_list)


def find_otsu_threshold(distances):
    distances = np.array(distances)

    hist, bin_edges = np.histogram(distances, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    current_max, threshold_idx = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0

    for i in range(len(hist)):
        sum_total += i * hist[i]

    for i in range(len(hist)):
        weight_background += hist[i]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += i * hist[i]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground

        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance_between > current_max:
            current_max = variance_between
            threshold_idx = i

    threshold_value = bin_centers[min(threshold_idx, len(bin_centers) - 1)]

    print(f"Otsu auto-detected threshold: {threshold_value:.4f}")
    return threshold_value


def find_gaussian_tail_threshold(distances, n_std: float = 2.0, fit_percentile: float = 30):
    distances = np.asarray(distances, dtype=float).ravel()
    if distances.size == 0:
        raise ValueError("`distances` must contain at least one element.")
    if not (0 < fit_percentile < 50):
        raise ValueError("`fit_percentile` should be between 0 and 50 (exclusive).")
    if n_std <= 0:
        raise ValueError("`n_std` must be positive.")

    tail_cut = np.percentile(distances, fit_percentile)
    fit_data = distances[distances <= tail_cut]
    if fit_data.size < 3:
        fit_data = distances

    mu, sigma = stats.norm.fit(fit_data)

    threshold = mu + n_std * sigma
    return float(threshold)
