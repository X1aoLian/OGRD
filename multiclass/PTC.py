
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import random
import os
import warnings

from load_dataset import create_train_test_split
from util import (
    GINSWDSingleModel,
    init_atoms_from_dataset,
    sample_anti_atoms_negatives,
    find_otsu_threshold,
    find_gaussian_tail_threshold
)
from torch_geometric.loader import DataLoader


def get_args():
    parser = argparse.ArgumentParser(
        description='Graph Open Set Recognition Training (Energy-Based)')

    parser.add_argument('--id_dataset', type=str, default='PTC_MR',
                        choices=['BZR', 'PTC-MR', 'AIDS', 'ENZYMES', 'IMDB-MULTI', 'Tox21', 'FreeSolv',
                                 'BBPB', 'ClinTox', 'Esol', ],
                        help='ID dataset name (known classes for training)')
    parser.add_argument('--ood_dataset', type=str, default='MUTAG',
                        choices=['COX2', 'MUTAG', 'DHFR', 'PROTEINS', 'IMDB-BINARY',
                                 'SIDER', 'ToxCast', 'BACE', 'LIPO', 'MUV', 'MSRC_9', 'MSRC_21'],
                        help='OOD dataset name (unknown classes for testing)')
    parser.add_argument('--id_source', type=str, default='TU', choices=['TU', 'MoleculeNet'],
                        help='Source for ID dataset')
    parser.add_argument('--ood_source', type=str, default='TU', choices=['TU', 'MoleculeNet'],
                        help='Source for OOD dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Train/test split ratio for ID dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='Dataset root directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GIN layers')
    parser.add_argument('--num_atom_supp', type=int, default=73,
                        help='Number of support atoms per class')
    parser.add_argument('--n_projections', type=int, default=50,
                        help='Number of SWD projections')
    parser.add_argument('--seed_swd', type=int, default=1997,
                        help='SWD random seed')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='MMD gamma parameter')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    parser.add_argument('--w_atom', type=float, default=1.0,
                        help='Atom distance loss weight')
    parser.add_argument('--w_pos', type=float, default=1.0,
                        help='Positive sample distance loss weight')
    parser.add_argument('--w_neg', type=float, default=1.0,
                        help='Negative sample distance loss weight')

    parser.add_argument('--margin', type=float, default=3,
                        help='Margin for negative samples (min distance to atoms)')
    parser.add_argument('--neg_ratio', type=float, default=0.8,
                        help='Ratio of negative samples to generate')

    parser.add_argument('--threshold_method', type=str, default='gaussian_tail',
                        choices=['otsu', 'gaussian_tail'],
                        help='Threshold detection method')
    parser.add_argument('--n_std', type=float, default=2.0,
                        help='Number of standard deviations for gaussian_tail method')

    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (epochs)')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    return parser.parse_args()


def compute_energy(distances):
    max_dist = torch.max(distances, dim=1, keepdim=True)[0]
    exp_terms = torch.exp(-(distances - max_dist))
    sum_exp = torch.sum(exp_terms, dim=1)
    energy = -(-max_dist.squeeze() + torch.log(sum_exp))
    return energy


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
    x_neg, neg_batch = sample_anti_atoms_negatives(model, num_neg)

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


def evaluate_energy_based_auto(model, test_data, num_known, device, threshold_method='gaussian_tail', n_std=2.0):
    model.eval()

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    all_energy_values = []
    all_distances_list = []
    all_labels = []

    print("Collecting test sample energy values...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            distances = model(data)
            energy_value = compute_energy(distances).item()
            all_energy_values.append(energy_value)
            all_distances_list.append(distances.cpu().numpy())
            all_labels.append(data.y.item())

    all_energy_values = np.array(all_energy_values)
    all_labels = np.array(all_labels)

    print(f"Collected energy values for {len(all_energy_values)} samples")
    print(f"Energy value range: [{np.min(all_energy_values):.6f}, {np.max(all_energy_values):.6f}]")

    if threshold_method == 'gaussian_tail':
        auto_threshold = find_gaussian_tail_threshold(all_energy_values, fit_percentile=45)
        method_name = f"Gaussian Tail Detection ({n_std} sigma)"
    else:
        auto_threshold = find_otsu_threshold(all_energy_values)
        method_name = "Otsu"

    print(f"Threshold detected by {method_name}: {auto_threshold:.6f}")

    all_preds = []
    print("Running predictions with energy values and detected threshold...")

    for i, (energy_value, distances) in enumerate(zip(all_energy_values, all_distances_list)):
        pred_class = np.argmin(distances)

        if energy_value > auto_threshold:
            pred_class = num_known

        all_preds.append(pred_class)

    all_preds = np.array(all_preds)

    binary_true = (all_labels == num_known).astype(int)
    binary_pred = (all_preds == num_known).astype(int)

    auc = roc_auc_score(binary_true, all_energy_values)

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
        known_accuracy = accuracy_score(known_true_labels, known_pred_labels)
    else:
        known_accuracy = 0.0

    overall_accuracy = accuracy_score(all_labels, all_preds)

    known_mask = all_labels != num_known
    unknown_mask = all_labels == num_known

    known_energies = all_energy_values[known_mask]
    unknown_energies = all_energy_values[unknown_mask]

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

    print(f"\n=== Energy Value Statistics ===")
    if len(known_energies) > 0:
        print(f"Known class mean energy: {np.mean(known_energies):.6f} +/- {np.std(known_energies):.6f}")
    if len(unknown_energies) > 0:
        print(f"Unknown class mean energy: {np.mean(unknown_energies):.6f} +/- {np.std(unknown_energies):.6f}")
    if len(known_energies) > 0 and len(unknown_energies) > 0:
        print(f"Energy separation: {np.mean(unknown_energies) - np.mean(known_energies):.6f}")

    return auc, binary_recall, binary_precision, overall_recall, overall_precision, known_accuracy, all_energy_values, all_labels, auto_threshold


def plot_energy_distribution_with_threshold(energy_values, labels, num_known, threshold,
                                            id_dataset, ood_dataset, save_path=None):
    plt.figure(figsize=(15, 8))

    known_mask = labels != num_known
    unknown_mask = labels == num_known

    known_energies = energy_values[known_mask]
    unknown_energies = energy_values[unknown_mask]

    all_energies = np.concatenate([known_energies, unknown_energies])
    min_energy, max_energy = np.min(all_energies), np.max(all_energies)
    bins = np.linspace(min_energy, max_energy, 51)

    plt.hist(known_energies, bins=bins, alpha=0.7,
             label=f'Known Classes ({id_dataset}, n={len(known_energies)})',
             color='blue', density=True)
    plt.hist(unknown_energies, bins=bins, alpha=0.7,
             label=f'Unknown Classes ({ood_dataset}, n={len(unknown_energies)})',
             color='red', density=True)

    if len(known_energies) > 0:
        known_mean = np.mean(known_energies)
        known_std = np.std(known_energies)
        plt.axvline(known_mean, color='blue', linestyle='--', alpha=0.8,
                    label=f'Known Mean: {known_mean:.3f}+/-{known_std:.3f}')

    if len(unknown_energies) > 0:
        unknown_mean = np.mean(unknown_energies)
        unknown_std = np.std(unknown_energies)
        plt.axvline(unknown_mean, color='red', linestyle='--', alpha=0.8,
                    label=f'Unknown Mean: {unknown_mean:.3f}+/-{unknown_std:.3f}')

    plt.axvline(threshold, color='green', linestyle='-', linewidth=3, alpha=0.9,
                label=f'Auto Threshold: {threshold:.3f}')

    plt.xlabel('Energy Value E(Hi)')
    plt.ylabel('Density')
    plt.title(f'Energy Distribution: {id_dataset} (Known) vs {ood_dataset} (Unknown)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n=== Parameter Configuration ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    print(f"\n=== Loading Dataset ===")
    print(f"ID dataset: {args.id_dataset} (source: {args.id_source})")
    print(f"OOD dataset: {args.ood_dataset} (source: {args.ood_source})")

    train_data, test_data, dataset_info = create_train_test_split(
        id_dataset_name=args.id_dataset,
        ood_dataset_name=args.ood_dataset,
        id_source=args.id_source,
        ood_source=args.ood_source,
        root=args.root,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    print(f"\n=== Dataset Information ===")
    print(f"Training set size: {dataset_info['train_size']}")
    print(f"Test set size: {dataset_info['test_size']}")
    print(f"  - ID test samples: {dataset_info['id_test_size']}")
    print(f"  - OOD test samples: {dataset_info['ood_test_size']}")
    print(f"Node feature dimension: {dataset_info['features']}")
    print(f"Number of ID classes: {dataset_info['id_classes']}")
    print(f"OOD label: {dataset_info['ood_label']}")

    num_known = dataset_info['id_classes']
    in_channels = dataset_info['features']

    initial_atoms = init_atoms_from_dataset(
        train_data,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=num_known,
        num_atom_supp=args.num_atom_supp,
        device=device
    )

    model = GINSWDSingleModel(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        initial_atoms=initial_atoms,
        n_projections=args.n_projections,
        seed=args.seed_swd,
        gamma=args.gamma
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n=== Starting Training ===")
    print(f"ID dataset: {args.id_dataset} -> OOD dataset: {args.ood_dataset}")
    print("Training objective: distance-based open set recognition (evaluated with energy values)")
    print("- Training: distance loss (positive samples close to atoms, negative samples far from atoms)")
    print("- Evaluation: energy value threshold detection")
    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        epoch_losses = train_epoch_distance_based(model, train_loader, optimizer, num_known, args, device)

        if epoch % args.eval_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | "
                  f"Atom={epoch_losses['atom']:.4f} | "
                  f"Pos={epoch_losses['pos']:.4f} | "
                  f"Neg={epoch_losses['neg']:.4f}")

    print(f"\n=== Starting Final Evaluation ===")
    auc, binary_recall, binary_precision, overall_recall, overall_precision, known_accuracy, all_energy_values, all_labels, auto_threshold = evaluate_energy_based_auto(
        model, test_data, num_known, device,
        threshold_method=args.threshold_method,
        n_std=args.n_std
    )

    print("\n=== Generating Energy Distribution Plot ===")
    save_path = f"{args.save_dir}/{args.id_dataset}_vs_{args.ood_dataset}_energy_distribution.png"
    plot_energy_distribution_with_threshold(
        all_energy_values, all_labels, num_known, auto_threshold,
        args.id_dataset, args.ood_dataset, save_path=save_path
    )

    return model, auc, binary_recall, binary_precision, overall_recall, overall_precision, known_accuracy, auto_threshold


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=== Graph Open Set Recognition Training (Distance Loss + Energy Evaluation) ===")
    print("Training uses distance loss, evaluation uses energy value threshold detection")
    print("Starting training and evaluation...")

    model, auc, binary_recall, binary_precision, overall_recall, overall_precision, known_accuracy, auto_threshold = train_and_evaluate(
        args)

    print(f"\n=== Final Results ===")
    print(f"ID dataset: {args.id_dataset}")
    print(f"OOD dataset: {args.ood_dataset}")
    print(f"Training method: distance-based loss")
    print(f"Evaluation method: energy value based")
    print(f"Threshold detection: {args.threshold_method}")
    print(f"Detected optimal threshold: {auto_threshold:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Binary Recall: {binary_recall:.4f}")
    print(f"Binary Precision: {binary_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Known class accuracy: {known_accuracy:.4f}")

    return model


if __name__ == '__main__':
    model = main()
