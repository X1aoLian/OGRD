#!/usr/bin/env python3
"""
修改后的第一个代码 - 添加AUC计算和5次运行统计
"""

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
    normalize_distances, load_dataset_unified, sample_neg_features_improved, find_gradient_descent_threshold
from util import find_otsu_threshold, find_gradient_gap_threshold
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ==================== 修改evaluate_distance_based_auto函数 ====================

def evaluate_distance_based_auto(model, test_known, all_unknown, batch_size, num_known, device,
                                 threshold_method='otsu'):
    """使用改进的阈值检测方法评估模型，添加AUC等完整指标"""
    model.eval()

    from torch.utils.data import ConcatDataset
    from torch_geometric.loader import DataLoader
    import torch

    test_data = ConcatDataset([test_known, all_unknown])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    all_min_distances = []

    # 收集所有距离
    print("🔍 收集测试样本距离...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            swd = model(data)
            min_distance = torch.min(swd, dim=1)[0].item()
            all_min_distances.append(min_distance)

    all_min_distances = np.array(all_min_distances)
    print(f"收集到 {len(all_min_distances)} 个样本的距离")
    print(f"距离范围: [{np.min(all_min_distances):.6f}, {np.max(all_min_distances):.6f}]")

    # 使用指定的阈值检测方法
    if threshold_method == 'gradient_descent':
        auto_threshold = find_gradient_descent_threshold(all_min_distances)
        method_name = "梯度下降检测"
    else:
        # 默认使用梯度下降
        method_name = "otsu"
        auto_threshold = find_otsu_threshold(all_min_distances)

    print(f"使用{method_name}检测到的阈值: {auto_threshold:.6f}")

    # 重新遍历进行预测
    all_preds = []
    all_labels = []

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print("🎯 使用检测阈值进行预测...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            swd = model(data)

            min_distance = all_min_distances[i]
            pred_class = torch.argmin(swd, dim=1).item()

            # 使用检测的阈值判断
            if min_distance > auto_threshold:
                pred_class = num_known  # unknown

            all_preds.append(pred_class)
            all_labels.append(data.y.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ========== 计算完整指标（参考第二个代码） ==========

    # 二分类指标 (Known vs Unknown)
    binary_true = (all_labels == num_known).astype(int)  # 1=Unknown, 0=Known
    binary_pred = (all_preds == num_known).astype(int)

    # AUC - 使用距离值作为score（距离越大，越可能是unknown）
    auc = roc_auc_score(binary_true, all_min_distances)

    # 二分类的recall和precision
    binary_recall = recall_score(binary_true, binary_pred)
    binary_precision = precision_score(binary_true, binary_pred)
    binary_f1 = f1_score(binary_true, binary_pred)

    # 整体的recall和precision（多分类）
    overall_recall = recall_score(all_labels, all_preds, average='macro')
    overall_precision = precision_score(all_labels, all_preds, average='macro')
    overall_f1 = f1_score(all_labels, all_preds, average='macro')

    # Known类准确率计算
    predicted_known_mask = (all_preds != num_known)
    if predicted_known_mask.sum() > 0:
        known_true_labels = all_labels[predicted_known_mask]
        known_pred_labels = all_preds[predicted_known_mask]
        known_accuracy = accuracy_score(known_true_labels, known_pred_labels)
    else:
        known_accuracy = 0.0

    # 整体准确率
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # 距离统计
    known_mask = all_labels != num_known
    unknown_mask = all_labels == num_known

    known_distances = all_min_distances[known_mask]
    unknown_distances = all_min_distances[unknown_mask]

    print(f"\n=== 🎯 {method_name}评估结果 ===")
    print(f"检测阈值: {auto_threshold:.6f}")
    print(f"AUC: {auc:.4f}")
    print(f"Binary Recall: {binary_recall:.4f}")
    print(f"Binary Precision: {binary_precision:.4f}")
    print(f"Binary F1: {binary_f1:.4f}")
    print(f"Overall Recall (Macro): {overall_recall:.4f}")
    print(f"Overall Precision (Macro): {overall_precision:.4f}")
    print(f"Overall F1 (Macro): {overall_f1:.4f}")
    print(f"Known类别准确率: {known_accuracy:.4f}")
    print(f"整体准确率: {overall_accuracy:.4f}")

    print(f"\n=== 📊 距离统计 ===")
    if len(known_distances) > 0:
        print(f"Known类最小距离: {np.mean(known_distances):.6f} ± {np.std(known_distances):.6f}")
    if len(unknown_distances) > 0:
        print(f"Unknown类最小距离: {np.mean(unknown_distances):.6f} ± {np.std(unknown_distances):.6f}")
    if len(known_distances) > 0 and len(unknown_distances) > 0:
        print(f"距离分离度: {np.mean(unknown_distances) - np.mean(known_distances):.6f}")

    # 返回主要指标
    return auc, binary_recall, binary_precision, binary_f1, known_accuracy, all_min_distances, all_labels, auto_threshold


# ==================== 其他函数保持不变 ====================

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Graph Open Set Recognition Training (Distance-Based with Auto Threshold) - Support Extended Datasets')

    # 数据相关参数 - 扩展数据集支持
    parser.add_argument('--dataset_name', type=str, default='MSRC_9',
                        choices=['MNIST', 'CLUSTER', 'CSL', 'CIFAR10', '10', '40',
                                 'MSRC_21', 'ENZYMES', 'IMDB-MULTI', 'COLLAB', 'MSRC_9', 'Synthie'],
                        help='Dataset name (now supports MNIST, CLUSTER, CSL, CIFAR10, ModelNet10/40)')
    parser.add_argument('--unknown_classes', type=int, nargs='+', default=[7, 8],
                        help='List of unknown class indices')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='Train/test split ratio')
    parser.add_argument('--root', type=str, default='../data',
                        help='Dataset root directory')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')

    # 新增数据集特定参数
    parser.add_argument('--scale_factor', type=float, default=1,
                        help='Data scaling factor (0-1) for fast experiments')
    parser.add_argument('--num_points', type=int, default=128,
                        help='Number of points to sample for ModelNet datasets')
    parser.add_argument('--k_neighbors', type=int, default=3,
                        help='Number of k-NN neighbors for ModelNet datasets')

    # 模型相关参数
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GIN layers')
    parser.add_argument('--num_atom_supp', type=int, default=40,
                        help='Number of support atoms per class')
    parser.add_argument('--n_projections', type=int, default=50,
                        help='Number of SWD projections')
    parser.add_argument('--seed_swd', type=int, default=1997,
                        help='SWD random seed')

    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # 损失函数权重
    parser.add_argument('--w_atom', type=float, default=1.0,
                        help='Atom distance loss weight')
    parser.add_argument('--w_pos', type=float, default=1.0,
                        help='Positive sample distance loss weight')
    parser.add_argument('--w_neg', type=float, default=1.0,
                        help='Negative sample distance loss weight')

    # 距离相关参数
    parser.add_argument('--margin', type=float, default=3,
                        help='Margin for negative samples (min distance to atoms)')
    parser.add_argument('--neg_ratio', type=float, default=0.8,
                        help='Ratio of negative samples to generate')

    # 可视化和评估参数
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (epochs)')

    # 输出相关参数
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    return parser.parse_args()


def compute_distance_based_loss(model, data, num_known, w_atom=1.0, w_pos=1.0, w_neg=1.0,
                                neg_ratio=0.2, margin=2.0):
    """
    基于距离的损失函数：
    - 正样本：接近对应atom
    - 负样本：远离所有atoms
    - 不使用熵值，直接用距离
    """
    device = data.x.device

    # 1. Atom分离损失 (保持不变)
    dist_mat = model.atom_distances()
    loss_atom = -torch.triu(dist_mat, 1).mean()

    # 2. 正样本距离损失
    swd_pos = model(data)  # [B, C]

    # 让每个正样本接近它的目标atom
    batch_size = swd_pos.size(0)
    target_distances = swd_pos[torch.arange(batch_size), data.y]  # [B]
    pos_loss = target_distances.mean()  # 最小化到目标atom的距离

    # 3. 负样本距离损失
    num_neg = max(1, int(batch_size * neg_ratio))
    x_neg, neg_batch = sample_anti_atoms_negatives(model, num_neg)

    neg_loss = torch.tensor(0.0, device=device)
    if len(x_neg) > 0:
        swd_neg = model(x_neg, neg_batch)  # [num_neg, C]

        # 让负样本远离所有atoms (最小距离也要大于margin)
        min_distances = torch.min(swd_neg, dim=1)[0]  # [num_neg]
        neg_loss = F.relu(margin - min_distances).mean()  # 距离小于margin时有惩罚

    return loss_atom, pos_loss, neg_loss


def train_epoch_distance_based(model, train_loader, optimizer, num_known, args, device):
    """基于距离的隔离训练"""
    model.train()

    atom_loss_total = 0
    pos_loss_total = 0
    neg_loss_total = 0

    for data in train_loader:
        data = data.to(device)

        # =================== 正样本训练 ===================
        optimizer.zero_grad()

        loss_atom, pos_loss, neg_loss = compute_distance_based_loss(
            model, data, num_known,
            w_atom=args.w_atom,
            w_pos=args.w_pos,
            w_neg=args.w_neg,
            neg_ratio=args.neg_ratio,
            margin=args.margin
        )

        # 只训练正样本和atom分离
        total_pos_loss = args.w_atom * loss_atom + args.w_pos * pos_loss
        total_pos_loss.backward()
        optimizer.step()

        # =================== 负样本训练 ===================
        optimizer.zero_grad()

        # 重新计算负样本损失 (确保干净梯度)
        _, _, neg_loss_clean = compute_distance_based_loss(
            model, data, num_known,
            w_atom=args.w_atom,
            w_pos=args.w_pos,
            w_neg=args.w_neg,
            neg_ratio=args.neg_ratio,
            margin=args.margin
        )

        # 只训练负样本
        total_neg_loss = args.w_neg * neg_loss_clean
        total_neg_loss.backward()
        optimizer.step()

        # 记录统计
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
    """使用距离损失训练并评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 统一数据加载（支持扩展数据集）
    train_known, test_known, all_unknown = load_dataset_unified(
        args.dataset_name, args.unknown_classes, args.ratio, args.root, args.seed,
        scale_factor=args.scale_factor,
        num_points=args.num_points,
        k_neighbors=args.k_neighbors
    )

    # 获取数据集信息
    sample_data = train_known[0]
    in_ch = sample_data.x.shape[1]

    # 确定已知类别数量
    if args.dataset_name == 'MNIST':
        all_classes = list(range(10))  # MNIST有0-9十个类别
    elif args.dataset_name == 'CLUSTER':
        all_classes = list(range(6))  # CLUSTER有6个类别
    elif args.dataset_name in ['CSL', 'CIFAR10']:
        all_classes = list(range(10))  # CSL和CIFAR10有10个类别
    elif args.dataset_name == '10':
        all_classes = list(range(10))  # ModelNet10有10个类别
    elif args.dataset_name == '40':
        all_classes = list(range(40))  # ModelNet40有40个类别
    elif args.dataset_name == 'COLORS-3':
        all_classes = list(range(11))  # COLORS-3有11个类别 (0-10)
    else:
        # 对于TU数据集，需要从实际数据中推断
        from torch_geometric.datasets import TUDataset
        tmp_dataset = TUDataset(args.root + '/TUDataset', args.dataset_name)
        all_classes = sorted(list(set([data.y.item() for data in tmp_dataset])))

    known_classes = [c for c in all_classes if c not in args.unknown_classes]
    num_known = len(known_classes)

    print(f"\n=== 📊 数据集信息 ===")
    print(f"数据集: {args.dataset_name}")
    print(f"总类别数: {len(all_classes)}")
    print(f"已知类别: {known_classes}")
    print(f"未知类别: {args.unknown_classes}")
    print(f"训练集大小: {len(train_known)}")
    print(f"测试集大小: {len(test_known)}")
    print(f"未知样本数: {len(all_unknown)}")
    print(f"节点特征维度: {in_ch}")

    # 初始化atoms (只包含已知类)
    initial_atoms = init_atoms_from_dataset(
        train_known,
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=num_known,  # 只有已知类
        num_atom_supp=args.num_atom_supp,
        device=device
    )

    # 创建模型
    model = GINSWDSingleModel(
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        initial_atoms=initial_atoms,
        n_projections=args.n_projections,
        seed=args.seed_swd
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_known, batch_size=args.batch_size, shuffle=True)

    # 创建保存目录
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n=== 🎯 使用基于距离的训练 + 自动阈值检测 ===")
    print(f"数据集: {args.dataset_name}")
    print("目标：正样本接近atoms，负样本远离atoms")
    print("特色：训练完成后自动检测最优阈值")
    print("开始训练...")

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        epoch_losses = train_epoch_distance_based(model, train_loader, optimizer, num_known, args, device)

        # 定期打印损失
        if epoch % args.eval_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | "
                  f"Atom={epoch_losses['atom']:.4f} | "
                  f"Pos={epoch_losses['pos']:.4f} | "
                  f"Neg={epoch_losses['neg']:.4f}")

    # 评估模型（使用自动阈值检测）
    print(f"\n=== 🔍 开始最终评估 (自动阈值检测) - {args.dataset_name} ===")
    auc, binary_recall, binary_precision, binary_f1, known_accuracy, all_distances, all_labels, auto_threshold = evaluate_distance_based_auto(
        model, test_known, all_unknown, args.batch_size, num_known, device
    )

    return model, auc, binary_recall, binary_precision, binary_f1, known_accuracy, auto_threshold


def main():
    """主函数 - 添加5次运行统计"""
    args = get_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== 🚀 Graph Open Set Recognition Training (Distance-Based + Auto Threshold + Extended Datasets) ===")
    print("✨ 新增支持: MNIST, CLUSTER, CSL, CIFAR10, ModelNet10/40数据集")
    print("🎯 使用直接距离损失，不依赖熵值计算")
    print("🔧 特色：自动检测最优阈值，无需手动调参")
    print("🔢 运行5次求平均值和标准差")
    print("开始训练和评估...")

    # ==================== 5次运行统计 ====================
    auc_list = []
    binary_f1_list = []
    binary_recall_list = []
    binary_precision_list = []
    known_accuracy_list = []

    for run_id in range(5):
        print(f"\n{'=' * 20} 🏃 第 {run_id + 1}/5 次运行 {'=' * 20}")

        # 每次运行使用不同的随机种子
        current_seed = args.seed + run_id
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        model, auc, binary_recall, binary_precision, binary_f1, known_accuracy, auto_threshold = train_and_evaluate(
            args)

        # 收集结果
        auc_list.append(auc)
        binary_f1_list.append(binary_f1)
        binary_recall_list.append(binary_recall)
        binary_precision_list.append(binary_precision)
        known_accuracy_list.append(known_accuracy)

        print(f"第{run_id + 1}次运行结果:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Binary F1: {binary_f1:.4f}")
        print(f"  Binary Recall: {binary_recall:.4f}")
        print(f"  Binary Precision: {binary_precision:.4f}")
        print(f"  Known Accuracy: {known_accuracy:.4f}")

    # ==================== 计算统计结果 ====================
    print(f"\n{'=' * 60}")
    print(f"🏆 最终统计结果 (5次运行)")
    print(f"{'=' * 60}")
    print(f"数据集: {args.dataset_name}")
    print(f"未知类别: {args.unknown_classes}")
    print(f"训练方法: 基于距离的损失")
    print(f"阈值检测: Otsu自动检测")

    # 计算平均值和标准差
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    f1_mean, f1_std = np.mean(binary_f1_list), np.std(binary_f1_list)
    recall_mean, recall_std = np.mean(binary_recall_list), np.std(binary_recall_list)
    precision_mean, precision_std = np.mean(binary_precision_list), np.std(binary_precision_list)
    known_acc_mean, known_acc_std = np.mean(known_accuracy_list), np.std(known_accuracy_list)

    print(f"\n📊 指标统计 (均值 ± 标准差):")
    print(f"AUC:              {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"Binary F1:        {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Binary Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"Binary Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Known Accuracy:   {known_acc_mean:.4f} ± {known_acc_std:.4f}")

    print(f"\n📈 详细数值 (便于复制):")
    print(f"AUC: {auc_mean:.6f}, {auc_std:.6f}")
    print(f"Binary F1: {f1_mean:.6f}, {f1_std:.6f}")
    print(f"Binary Recall: {recall_mean:.6f}, {recall_std:.6f}")
    print(f"Binary Precision: {precision_mean:.6f}, {precision_std:.6f}")
    print(f"Known Accuracy: {known_acc_mean:.6f}, {known_acc_std:.6f}")

    return auc_list, binary_f1_list, binary_recall_list, binary_precision_list, known_accuracy_list


if __name__ == '__main__':
    # 示例运行命令（命令行使用）：
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name MNIST --unknown_classes 9 --epochs 30 --scale_factor 0.01
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name CLUSTER --unknown_classes 5 --epochs 50 --scale_factor 0.05
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name CSL --unknown_classes 8 9 --epochs 50
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name CIFAR10 --unknown_classes 9 --epochs 50 --scale_factor 0.01
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name 10 --unknown_classes 8 9 --epochs 100 --scale_factor 0.1
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name 40 --unknown_classes 35 36 37 38 39 --epochs 150 --scale_factor 0.1
    # python train_distance_based_auto_with_extended_datasets.py --dataset_name ENZYMES --unknown_classes 5 --epochs 200

    # PyCharm右键直接运行时会使用默认MNIST配置
    results = main()