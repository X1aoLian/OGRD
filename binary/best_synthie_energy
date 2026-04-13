import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, \
    precision_score
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MNISTSuperpixels, GNNBenchmarkDataset, ModelNet
from torch_geometric.transforms import OneHotDegree, SamplePoints, KNNGraph, FaceToEdge, Compose
from torch_geometric.utils import degree
from torch.utils.data import Subset
import random

from util import find_first_valley_threshold, init_atoms_from_dataset, GINSWDSingleModel, sample_anti_atoms_negatives, \
    normalize_distances, load_dataset_unified, sample_neg_features_improved, find_gradient_descent_threshold, \
    sample_neg_features_atoms, find_gaussian_tail_threshold
from util import find_otsu_threshold, find_gradient_gap_threshold
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def evaluate_distance_based_auto(model, test_known, all_unknown, batch_size, num_known, device,
                                 threshold_method='valley'):
    model.eval()

    from torch.utils.data import ConcatDataset
    from torch_geometric.loader import DataLoader
    import torch

    test_data = ConcatDataset([test_known, all_unknown])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    all_min_distances = []

    print("Collecting test sample distances...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            swd = model(data)
            min_distance = torch.min(swd, dim=1)[0].item()
            all_min_distances.append(min_distance)

    all_min_distances = np.array(all_min_distances)
    print(f"Collected distances for {len(all_min_distances)} samples")
    print(f"Distance range: [{np.min(all_min_distances):.6f}, {np.max(all_min_distances):.6f}]")

    if threshold_method == 'gradient_descent':
        auto_threshold = find_gradient_descent_threshold(all_min_distances)
        method_name = "Gradient Descent Detection"
    elif threshold_method == 'valley':
        auto_threshold = find_gaussian_tail_threshold(all_min_distances)
        method_name = "Second Derivative Detection"
    else:
        method_name = "otsu"
        auto_threshold = find_otsu_threshold(all_min_distances)

    print(f"Threshold detected by {method_name}: {auto_threshold:.6f}")

    all_preds = []
    all_labels = []

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print("Running predictions with detected threshold...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            swd = model(data)

            min_distance = all_min_distances[i]
            pred_class = torch.argmin(swd, dim=1).item()

            if min_distance > auto_threshold:
                pred_class = num_known

            all_preds.append(pred_class)
            all_labels.append(data.y.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_true = (all_labels == num_known).astype(int)
    binary_pred = (all_preds == num_known).astype(int)

    auc = roc_auc_score(binary_true, all_min_distances)

    binary_recall = recall_score(binary_true, binary_pred)
    binary_precision = precision_score(binary_true, binary_pred)
    binary_f1 = f1_score(binary_true, binary_pred)

    overall_recall = recall_score(all_labels, all_preds, average='macro')
    overall_precision = precision_score(all_labels, all_preds, average='macro')
    overall_f1 = f1_score(all_labels, all_preds, average='macro')

    predicted_known_mask = (all_preds != num_known)
    if predicted_known_mask.sum() > 0:
        known_true_labels = all_labels[predicted_known_mask]
        known_pred_labels = all_preds[predicted_known_mask]
        print(known_true_labels)
        print(known_pred_labels)
        known_accuracy = accuracy_score(known_true_labels, known_pred_labels)
    else:
        known_accuracy = 0.0

    overall_accuracy = accuracy_score(all_labels, all_preds)

    known_mask = all_labels != num_known
    unknown_mask = all_labels == num_known

    known_distances = all_min_distances[known_mask]
    unknown_distances = all_min_distances[unknown_mask]

    print(f"\n=== {method_name} Evaluation Results ===")
    print(f"Detection threshold: {auto_threshold:.6f}")
    print(f"AUC: {auc:.4f}")
    print(f"Binary Recall: {binary_recall:.4f}")
    print(f"Binary Precision: {binary_precision:.4f}")
    print(f"Binary F1: {binary_f1:.4f}")
    print(f"Overall Recall (Macro): {overall_recall:.4f}")
    print(f"Overall Precision (Macro): {overall_precision:.4f}")
    print(f"Overall F1 (Macro): {overall_f1:.4f}")
    print(f"Known class accuracy: {known_accuracy:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    print(f"\n=== Distance Statistics ===")
    if len(known_distances) > 0:
        print(f"Known class min distance: {np.mean(known_distances):.6f} +/- {np.std(known_distances):.6f}")
    if len(unknown_distances) > 0:
        print(f"Unknown class min distance: {np.mean(unknown_distances):.6f} +/- {np.std(unknown_distances):.6f}")
    if len(known_distances) > 0 and len(unknown_distances) > 0:
        print(f"Distance separation: {np.mean(unknown_distances) - np.mean(known_distances):.6f}")

    return auc, binary_recall, binary_precision, binary_f1, known_accuracy, all_min_distances, all_labels, auto_threshold


def get_args():
    parser = argparse.ArgumentParser(
        description='Graph Open Set Recognition Training (Distance-Based with Auto Threshold) - Support Extended Datasets')
    parser.add_argument('--dataset_name', type=str, default='COLLAB',
                        choices=['MNIST', 'CLUSTER', 'CSL', 'CIFAR10', '10', '40',
                                 'MSRC_21', 'ENZYMES', 'IMDB-MULTI', 'COLLAB', 'MSRC_9', 'Synthie'],
                        help='Dataset name (now supports MNIST, CLUSTER, CSL, CIFAR10, ModelNet10/40)')
    parser.add_argument('--unknown_classes', type=int, nargs='+', default=[2],
                        help='List of unknown class indices')
    parser.add_argument('--ratio', type=float, default=0.9,
                        help='Train/test split ratio')
    parser.add_argument('--root', type=str, default='../data',
                        help='Dataset root directory')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')

    parser.add_argument('--scale_factor', type=float, default=0.1,
                        help='Data scaling factor (0-1) for fast experiments')
    parser.add_argument('--num_points', type=int, default=128,
                        help='Number of points to sample for ModelNet datasets')
    parser.add_argument('--k_neighbors', type=int, default=3,
                        help='Number of k-NN neighbors for ModelNet datasets')

    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GIN layers')
    parser.add_argument('--num_atom_supp', type=int, default=100,
                        help='Number of support atoms per class')
    parser.add_argument('--n_projections', type=int, default=50,
                        help='Number of SWD projections')
    parser.add_argument('--seed_swd', type=int, default=1997,
                        help='SWD random seed')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    parser.add_argument('--w_atom', type=float, default=0.14054584785805668,
                        help='Atom distance loss weight')
    parser.add_argument('--w_pos', type=float, default=3.1218937952847026,
                        help='Positive sample distance loss weight')
    parser.add_argument('--w_neg', type=float, default=5.069747811940598,
                        help='Negative sample distance loss weight')

    parser.add_argument('--margin', type=float, default=3,
                        help='Margin for negative samples (min distance to atoms)')
    parser.add_argument('--neg_ratio', type=float, default=0.6,
                        help='Ratio of negative samples to generate')

    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (epochs)')

    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    return parser.parse_args()


def compute_distance_based_loss(model, data, num_known, w_atom=1.0, w_pos=1.0, w_neg=1.0,
                                neg_ratio=0.2, margin=2.0):
    device = data.x.device

    dist_mat = model.atom_distances()
    loss_atom = -torch.triu(dist_mat, 1).mean()

    swd_pos = model(data)

    batch_size = swd_pos.size(0)
    target_distances = swd_pos[torch.arange(batch_size), data.y]
    pos_loss = target_distances.mean()

    num_neg = max(1, int(batch_size * neg_ratio))
    x_neg, neg_batch = sample_neg_features_atoms(model, data, ratio=neg_ratio)

    neg_loss = torch.tensor(0.0, device=device)
    if len(x_neg) > 0:
        swd_neg = model(x_neg, neg_batch)

        min_distances = torch.min(swd_neg, dim=1)[0]
        neg_loss = F.relu(margin - min_distances).mean()

    return loss_atom, pos_loss, neg_loss


def train_epoch_distance_based(model, train_loader, optimizer, num_known, args, device):
    model.train()

    atom_loss_total = 0
    pos_loss_total = 0
    neg_loss_total = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        loss_atom, pos_loss, neg_loss = compute_distance_based_loss(
            model, data, num_known,
            w_atom=args.w_atom,
            w_pos=args.w_pos,
            w_neg=args.w_neg,
            neg_ratio=args.neg_ratio,
            margin=args.margin
        )

        total_pos_loss = args.w_atom * loss_atom + args.w_pos * pos_loss
        total_pos_loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        _, _, neg_loss_clean = compute_distance_based_loss(
            model, data, num_known,
            w_atom=args.w_atom,
            w_pos=args.w_pos,
            w_neg=args.w_neg,
            neg_ratio=args.neg_ratio,
            margin=args.margin
        )

        total_neg_loss = args.w_neg * neg_loss_clean
        total_neg_loss.backward()
        optimizer.step()

        atom_loss_total += loss_atom.item()
        pos_loss_total += pos_loss.item()
        neg_loss_total += neg_loss.item()

    n_batches = len(train_loader)
    return {
        'atom': atom_loss_total / n_batches,
        'pos': pos_loss_total / n_batches,
        'neg': neg_loss_total / n_batches
    }


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_known, test_known, all_unknown = load_dataset_unified(
        args.dataset_name, args.unknown_classes, args.ratio, args.root, args.seed,
        scale_factor=args.scale_factor,
        num_points=args.num_points,
        k_neighbors=args.k_neighbors
    )

    sample_data = train_known[0]
    in_ch = sample_data.x.shape[1]

    if args.dataset_name == 'MNIST':
        all_classes = list(range(10))
    elif args.dataset_name == 'CLUSTER':
        all_classes = list(range(6))
    elif args.dataset_name in ['CSL', 'CIFAR10']:
        all_classes = list(range(10))
    elif args.dataset_name == '10':
        all_classes = list(range(10))
    elif args.dataset_name == '40':
        all_classes = list(range(40))
    elif args.dataset_name == 'COLORS-3':
        all_classes = list(range(11))
    else:
        from torch_geometric.datasets import TUDataset
        tmp_dataset = TUDataset(args.root + '/TUDataset', args.dataset_name)
        all_classes = sorted(list(set([data.y.item() for data in tmp_dataset])))

    known_classes = [c for c in all_classes if c not in args.unknown_classes]
    num_known = len(known_classes)

    print(f"\n=== Dataset Information ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Total number of classes: {len(all_classes)}")
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {args.unknown_classes}")
    print(f"Training set size: {len(train_known)}")
    print(f"Test set size: {len(test_known)}")
    print(f"Number of unknown samples: {len(all_unknown)}")
    print(f"Node feature dimension: {in_ch}")

    initial_atoms = init_atoms_from_dataset(
        train_known,
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=num_known,
        num_atom_supp=args.num_atom_supp,
        device=device
    )

    model = GINSWDSingleModel(
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        initial_atoms=initial_atoms,
        n_projections=args.n_projections,
        seed=args.seed_swd
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_known, batch_size=args.batch_size, shuffle=True)

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n=== Distance-Based Training + Auto Threshold Detection ===")
    print(f"Dataset: {args.dataset_name}")
    print("Goal: positive samples close to atoms, negative samples far from atoms")
    print("Feature: automatic optimal threshold detection after training")
    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        epoch_losses = train_epoch_distance_based(model, train_loader, optimizer, num_known, args, device)

        if epoch % args.eval_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | "
                  f"Atom={epoch_losses['atom']:.4f} | "
                  f"Pos={epoch_losses['pos']:.4f} | "
                  f"Neg={epoch_losses['neg']:.4f}")

    print(f"\n=== Starting Final Evaluation (Auto Threshold Detection) - {args.dataset_name} ===")
    auc, binary_recall, binary_precision, binary_f1, known_accuracy, all_distances, all_labels, auto_threshold = evaluate_distance_based_auto(
        model, test_known, all_unknown, args.batch_size, num_known, device
    )

    return model, auc, binary_recall, binary_precision, binary_f1, known_accuracy, auto_threshold


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== Graph Open Set Recognition Training (Distance-Based + Auto Threshold + Extended Datasets) ===")
    print("New support: MNIST, CLUSTER, CSL, CIFAR10, ModelNet10/40 datasets")
    print("Uses direct distance loss, no entropy computation required")
    print("Feature: automatic optimal threshold detection, no manual tuning")
    print("Running 5 times to compute mean and standard deviation")
    print("Starting training and evaluation...")

    auc_list = []
    binary_f1_list = []
    binary_recall_list = []
    binary_precision_list = []
    known_accuracy_list = []

    for run_id in range(1):
        print(f"\n{'=' * 20} Run {run_id + 1}/5 {'=' * 20}")

        current_seed = args.seed + run_id
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        model, auc, binary_recall, binary_precision, binary_f1, known_accuracy, auto_threshold = train_and_evaluate(
            args)

        auc_list.append(auc)
        binary_f1_list.append(binary_f1)
        binary_recall_list.append(binary_recall)
        binary_precision_list.append(binary_precision)
        known_accuracy_list.append(known_accuracy)

        print(f"Run {run_id + 1} results:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Binary F1: {binary_f1:.4f}")
        print(f"  Binary Recall: {binary_recall:.4f}")
        print(f"  Binary Precision: {binary_precision:.4f}")
        print(f"  Known Accuracy: {known_accuracy:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Final Statistics (5 Runs)")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Unknown classes: {args.unknown_classes}")
    print(f"Training method: distance-based loss")
    print(f"Threshold detection: Otsu automatic")

    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    f1_mean, f1_std = np.mean(binary_f1_list), np.std(binary_f1_list)
    recall_mean, recall_std = np.mean(binary_recall_list), np.std(binary_recall_list)
    precision_mean, precision_std = np.mean(binary_precision_list), np.std(binary_precision_list)
    known_acc_mean, known_acc_std = np.mean(known_accuracy_list), np.std(known_accuracy_list)

    print(f"\nMetric Statistics (mean +/- std):")
    print(f"AUC:              {auc_mean:.4f} +/- {auc_std:.4f}")
    print(f"Binary F1:        {f1_mean:.4f} +/- {f1_std:.4f}")
    print(f"Binary Recall:    {recall_mean:.4f} +/- {recall_std:.4f}")
    print(f"Binary Precision: {precision_mean:.4f} +/- {precision_std:.4f}")
    print(f"Known Accuracy:   {known_acc_mean:.4f} +/- {known_acc_std:.4f}")

    print(f"\nDetailed values (for copy-paste):")
    print(f"AUC: {auc_mean:.6f}, {auc_std:.6f}")
    print(f"Binary F1: {f1_mean:.6f}, {f1_std:.6f}")
    print(f"Binary Recall: {recall_mean:.6f}, {recall_std:.6f}")
    print(f"Binary Precision: {precision_mean:.6f}, {precision_std:.6f}")
    print(f"Known Accuracy: {known_acc_mean:.6f}, {known_acc_std:.6f}")

    return auc_list, binary_f1_list, binary_recall_list, binary_precision_list, known_accuracy_list


if __name__ == '__main__':
    results = main()
