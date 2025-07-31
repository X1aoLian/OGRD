# util.py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch_geometric.datasets import TUDataset, MNISTSuperpixels
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import Subset
from torch_geometric.utils import degree
import torch
import random
import torch.nn.functional as F


def split_dataset(
        dataset_name: str,
        unknown_classes: list,
        ratio: float,
        root: str = 'data/TUDataset',
        seed: int = 42,
        scale_factor: float = 1.0
):
    """
    TU数据集分割函数 - 支持scale_factor缩放

    Args:
        dataset_name: TU数据集名称
        unknown_classes: 未知类别列表
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        scale_factor: 数据缩放因子（0-1之间）

    Returns:
        train_known, test_known, all_unknown: 三个数据集
    """
    import random
    print('---------------------------------------------------------------')
    # 1. 先加载一次不带 transform 的数据集，用于计算 max_degree
    tmp = TUDataset(root, dataset_name,use_node_attr=True)


    # 2. 如果没有节点特征，则计算 max_degree 并重新加载带 transform 的数据集
    if tmp.num_node_features == 0:
        max_deg = 0
        for data in tmp:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_deg = max(max_deg, int(deg.max()))
        dataset = TUDataset(root, dataset_name, transform=OneHotDegree(max_deg))
    else:
        dataset = tmp

    # 3. 收集所有标签
    labels = torch.tensor([int(data.y) for data in dataset])

    # 4. 获取所有唯一类别并创建标签映射
    all_classes = sorted(labels.unique().tolist())
    known_classes = [c for c in all_classes if c not in unknown_classes]

    # 创建从原始标签到新标签的映射
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")
    print(f"数据缩放因子: {scale_factor}")

    # 5. 数据缩放逻辑
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放TU数据集 (缩放因子: {scale_factor}) ===")

        # 按类别分组所有样本
        all_class_indices = {}
        for idx in range(len(dataset)):
            label = labels[idx].item()
            if label not in all_class_indices:
                all_class_indices[label] = []
            all_class_indices[label].append(idx)

        # 对每个类别等比例采样
        scaled_indices = []
        for class_label in all_classes:
            if class_label in all_class_indices:
                class_indices = all_class_indices[class_label]
                original_count = len(class_indices)
                scaled_count = max(1, int(original_count * scale_factor))  # 至少保留1个样本

                # 随机采样
                random.shuffle(class_indices)
                sampled_indices = class_indices[:scaled_count]
                scaled_indices.extend(sampled_indices)

                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        # 重新构建缩放后的数据集和标签
        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        # 更新数据集和标签
        dataset = scaled_dataset
        labels = scaled_labels

        print(f"缩放后数据集大小: {len(dataset)}")

    # 6. 划分已知/未知
    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 7. 分层采样 - 按类别分别划分train/test
    # 按类别分组已知样本的索引
    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:  # 只处理已知类别
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== TU数据集分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:  # 添加检查，防止缩放后某些类别消失
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train

            # 随机打乱该类别的样本
            random.shuffle(indices)

            # 分配到训练集和测试集
            class_train = indices[:n_train]
            class_test = indices[n_train:]

            train_idx.extend(class_train)
            test_idx.extend(class_test)

            print(f"原始类别 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    # 8. 打乱最终的训练和测试索引
    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 9. 调整未知样本数量以匹配测试集大小
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        # 如果未知样本太多，随机采样
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        # 如果未知样本太少，保持原有数量但给出提示
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 10. 创建重新映射标签的数据集包装器
    class RelabeledSubset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]
            # 重新映射标签
            if self.is_unknown:
                # 所有未知类别统一映射为 num_known
                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
            else:
                # 安全地处理标签转换
                try:
                    if data.y.numel() == 1:
                        original_label = data.y.item()
                    else:
                        # 如果是多维张量，取第一个元素
                        original_label = data.y.flatten()[0].item()
                except:
                    # 最后的备选方案
                    original_label = int(data.y.cpu().numpy().flatten()[0])

                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    # 11. 构造返回的数据集
    num_known = len(known_classes)

    train_known = RelabeledSubset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledSubset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledSubset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    # 12. 验证分层采样效果
    print(f"\n=== 验证TU数据集训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:  # 添加安全检查
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_dataset_with_class_merging(
        dataset_name: str,
        class_mapping: dict,  # {new_label: [old_labels]} 例如 {0: [0,1], 1: [2,3], 'unknown': [4]}
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        **kwargs
):
    """
    支持类别合并的数据集分割函数

    Args:
        dataset_name: 数据集名称
        class_mapping: 类别映射字典，格式为 {new_label: [old_labels]}
                      例如：{0: [0,1], 1: [2,3], 'unknown': [4]}
                      其中 'unknown' 键对应的类别将被标记为未知类别
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        scale_factor: 数据缩放因子（0-1之间）
        **kwargs: 额外参数（传递给原始加载函数）

    Returns:
        train_known, test_known, all_unknown: 三个数据集
    """
    import random
    import torch

    print(f"=== 🔄 类别合并数据集分割：{dataset_name} ===")
    print(f"类别映射方案: {class_mapping}")

    # 1. 验证class_mapping格式
    if 'unknown' not in class_mapping:
        raise ValueError("class_mapping必须包含'unknown'键来指定未知类别")

    # 2. 提取unknown_classes和构建label_mapping
    unknown_classes = class_mapping['unknown']

    # 构建已知类别的映射
    known_mapping = {}
    for new_label, old_labels in class_mapping.items():
        if new_label != 'unknown':
            for old_label in old_labels:
                known_mapping[old_label] = new_label

    # 获取所有涉及的原始类别
    all_involved_classes = []
    for old_labels in class_mapping.values():
        all_involved_classes.extend(old_labels)

    print(f"原始未知类别: {unknown_classes}")
    print(f"已知类别映射: {known_mapping}")
    print(f"涉及的所有原始类别: {sorted(all_involved_classes)}")

    # 3. 使用原始函数加载数据集
    # 这里我们先用空的unknown_classes来获取完整数据，然后手动处理
    temp_train, temp_test, temp_unknown = load_dataset_unified(
        dataset_name, [], ratio, root, seed, scale_factor=scale_factor, **kwargs
    )

    # 合并所有数据
    all_data_indices = list(range(len(temp_train) + len(temp_test) + len(temp_unknown)))
    all_dataset = []
    all_labels = []

    # 收集所有数据和原始标签
    for dataset_part in [temp_train, temp_test, temp_unknown]:
        for data in dataset_part:
            all_dataset.append(data)
            all_labels.append(data.y.item())

    print(f"收集到总数据量: {len(all_dataset)}")
    print(f"原始标签分布: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    # 4. 过滤：只保留在class_mapping中定义的类别
    filtered_dataset = []
    filtered_labels = []

    for i, (data, label) in enumerate(zip(all_dataset, all_labels)):
        if label in all_involved_classes:
            filtered_dataset.append(data)
            filtered_labels.append(label)

    print(f"过滤后数据量: {len(filtered_dataset)}")
    print(f"过滤后标签分布: {dict(zip(*np.unique(filtered_labels, return_counts=True)))}")

    # 5. 应用类别合并映射
    merged_dataset = []
    merged_labels = []

    for data, old_label in zip(filtered_dataset, filtered_labels):
        # 创建数据副本
        new_data = data.clone()

        # 确定新标签
        if old_label in unknown_classes:
            # 未知类别：标记为特殊值，稍后会重新设置
            new_label = -1  # 临时标记
        else:
            # 已知类别：根据映射转换
            if old_label in known_mapping:
                new_label = known_mapping[old_label]
            else:
                print(f"警告：标签 {old_label} 不在已知映射中，跳过")
                continue

        new_data.y = torch.tensor(new_label, dtype=data.y.dtype, device=data.y.device)
        merged_dataset.append(new_data)
        merged_labels.append(new_label)

    # 6. 统计合并后的标签分布
    known_labels = [label for label in merged_labels if label != -1]
    unknown_count = sum(1 for label in merged_labels if label == -1)

    print(f"\n=== 📊 类别合并结果 ===")
    if known_labels:
        known_distribution = dict(zip(*np.unique(known_labels, return_counts=True)))
        print(f"已知类别分布: {known_distribution}")
    print(f"未知类别数量: {unknown_count}")

    # 确定最终的已知类别数量
    num_known = len(set(known_labels)) if known_labels else 0

    # 7. 重新设置未知类别标签为num_known
    for data, label in zip(merged_dataset, merged_labels):
        if label == -1:
            data.y = torch.tensor(num_known, dtype=data.y.dtype, device=data.y.device)

    # 更新merged_labels
    merged_labels = [num_known if label == -1 else label for label in merged_labels]

    # 8. 数据缩放（如果需要）
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 🔄 等比例缩放数据集 (缩放因子: {scale_factor}) ===")

        # 按新标签分组
        class_indices = {}
        for idx, label in enumerate(merged_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # 对每个类别等比例采样
        scaled_indices = []
        for class_label, indices in class_indices.items():
            original_count = len(indices)
            scaled_count = max(1, int(original_count * scale_factor))

            random.shuffle(indices)
            sampled_indices = indices[:scaled_count]
            scaled_indices.extend(sampled_indices)

            class_name = f"Class {class_label}" if class_label != num_known else "Unknown"
            print(f"{class_name}: {original_count} -> {scaled_count} 样本")

        # 重新构建缩放后的数据集
        scaled_dataset = [merged_dataset[i] for i in scaled_indices]
        scaled_labels = [merged_labels[i] for i in scaled_indices]

        merged_dataset = scaled_dataset
        merged_labels = scaled_labels

        print(f"缩放后数据集大小: {len(merged_dataset)}")

    # 9. 分离已知和未知样本
    unknown_indices = [i for i, label in enumerate(merged_labels) if label == num_known]
    known_indices = [i for i, label in enumerate(merged_labels) if label != num_known]

    # 10. 对已知类别进行分层采样
    class_indices = {}
    for idx in known_indices:
        label = merged_labels[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== 🎯 分层采样详情 ===")
    for class_label in sorted(class_indices.keys()):
        indices = class_indices[class_label]
        n_samples = len(indices)
        n_train = int(n_samples * ratio)
        n_test = n_samples - n_train

        random.shuffle(indices)
        class_train = indices[:n_train]
        class_test = indices[n_train:]

        train_idx.extend(class_train)
        test_idx.extend(class_test)

        print(f"合并后类别 {class_label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    # 11. 调整未知样本数量
    random.shuffle(train_idx)
    random.shuffle(test_idx)
    random.shuffle(unknown_indices)

    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_indices)}")

    if len(unknown_indices) > test_known_size:
        unknown_indices = unknown_indices[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_indices) < test_known_size:
        print(f"警告：未知样本数量({len(unknown_indices)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_indices)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_indices)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_indices)}")

    # 12. 创建最终的数据集
    class MergedDatasetSubset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_known = MergedDatasetSubset(merged_dataset, train_idx)
    test_known = MergedDatasetSubset(merged_dataset, test_idx)
    all_unknown = MergedDatasetSubset(merged_dataset, unknown_indices)

    # 13. 验证最终结果
    print(f"\n=== ✅ 验证合并后训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        label = merged_labels[idx]
        train_label_count[label] = train_label_count.get(label, 0) + 1

    print(f"训练集标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


# 使用示例函数
def example_usage():
    """使用示例"""

    # 示例1：将MNIST的10个数字合并为3类
    class_mapping_mnist = {
        0: [0, 1],  # 数字0,1 合并为类别0
        1: [2, 3, 4],  # 数字2,3,4 合并为类别1
        2: [5, 6, 7],  # 数字5,6,7 合并为类别2
        'unknown': [8, 9]  # 数字8,9 作为未知类别
    }

    # 示例2：将5类数据集合并为2类
    class_mapping_5to2 = {
        0: [0, 1],  # 原始类别0,1 合并为新类别0
        1: [2, 3],  # 原始类别2,3 合并为新类别1
        'unknown': [4]  # 原始类别4 作为未知类别
    }

    # 示例3：将10类数据集合并为3类
    class_mapping_10to3 = {
        0: [0, 1, 2],  # 前3类合并
        1: [3, 4, 5],  # 中3类合并
        2: [6, 7],  # 后2类合并
        'unknown': [8, 9]  # 最后2类作为未知
    }

    print("=== 🧪 使用示例 ===")
    print("示例1 - MNIST 10→3类:")
    print(f"  class_mapping = {class_mapping_mnist}")
    print("\n示例2 - 5类→2类:")
    print(f"  class_mapping = {class_mapping_5to2}")
    print("\n示例3 - 10类→3类:")
    print(f"  class_mapping = {class_mapping_10to3}")


def split_mnist_dataset(unknown_classes, ratio, root='../data', seed=42, scale_factor=0.01):
    """
    专门为MNIST数据集设计的分割函数

    Args:
        unknown_classes: 未知类别列表
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        scale_factor: 缩放因子，0-1之间，用于等比例缩小数据集
                     例如：0.1表示只使用10%的数据，0.5表示使用50%的数据
    """
    import random
    import torch

    # 加载MNIST数据集
    train_dataset = MNISTSuperpixels(root=root + '/MNISTSuperpixels', train=True)
    test_dataset = MNISTSuperpixels(root=root + '/MNISTSuperpixels', train=False)

    # 合并训练和测试数据，然后重新分割
    all_dataset = list(train_dataset) + list(test_dataset)

    print(f"MNIST数据集原始总大小: {len(all_dataset)}")
    print(f"节点特征维度: {all_dataset[0].x.shape[1]}")
    print(f"总类别数: 10 (数字0-9)")
    print(f"数据缩放因子: {scale_factor}")

    # 收集所有标签
    labels = torch.tensor([int(data.y) for data in all_dataset])

    # 获取所有唯一类别并创建标签映射
    all_classes = list(range(10))  # MNIST有0-9十个类别
    known_classes = [c for c in all_classes if c not in unknown_classes]

    # 创建从原始标签到新标签的映射
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")

    # 如果需要缩放数据集，先按类别等比例采样
    random.seed(seed)

    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放数据集 (缩放因子: {scale_factor}) ===")

        # 按类别分组所有样本
        all_class_indices = {}
        for idx in range(len(all_dataset)):
            label = labels[idx].item()
            if label not in all_class_indices:
                all_class_indices[label] = []
            all_class_indices[label].append(idx)

        # 对每个类别等比例采样
        scaled_indices = []
        for class_label in all_classes:
            if class_label in all_class_indices:
                class_indices = all_class_indices[class_label]
                original_count = len(class_indices)
                scaled_count = max(1, int(original_count * scale_factor))  # 至少保留1个样本

                # 随机采样
                random.shuffle(class_indices)
                sampled_indices = class_indices[:scaled_count]
                scaled_indices.extend(sampled_indices)

                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        # 重新构建缩放后的数据集和标签
        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        # 更新数据集和标签
        all_dataset = scaled_dataset
        labels = scaled_labels

        print(f"缩放后数据集大小: {len(all_dataset)}")

    # 分离已知和未知样本
    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 按类别分组已知样本的索引
    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:  # 只处理已知类别
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    # 分层采样
    train_idx = []
    test_idx = []

    print(f"\n=== MNIST分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:  # 确保该类别有样本
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train

            # 随机打乱该类别的样本
            random.shuffle(indices)

            # 分配到训练集和测试集
            class_train = indices[:n_train]
            class_test = indices[n_train:]

            train_idx.extend(class_train)
            test_idx.extend(class_test)

            print(f"数字 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    # 打乱最终的训练和测试索引
    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 调整未知样本数量以匹配测试集大小
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        # 如果未知样本太多，随机采样
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        # 如果未知样本太少，保持原有数量但给出提示
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 创建重新映射标签的数据集包装器
    class RelabeledMNISTSubset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]
            # 重新映射标签
            if self.is_unknown:
                # 所有未知类别统一映射为 num_known

                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
                # 或

            else:
                # 已知类别重新映射
                original_label = data.y.item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    # 构造返回的数据集
    num_known = len(known_classes)

    train_known = RelabeledMNISTSubset(all_dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledMNISTSubset(all_dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledMNISTSubset(all_dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    # 验证分层采样效果
    print(f"\n=== 验证MNIST训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_colors3_dataset(
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0
):
    """
    COLORS-3数据集专用分割函数

    Args:
        unknown_classes: 未知类别列表
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        scale_factor: 数据缩放因子（0-1之间）

    Returns:
        train_known, test_known, all_unknown: 三个数据集
    """
    import random
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import OneHotDegree
    from torch_geometric.utils import degree

    dataset_name = 'COLORS-3'

    # 1. 加载COLORS-3数据集
    tmp = TUDataset(root + '/TUDataset', dataset_name,use_node_attr=True)

    # 2. 如果没有节点特征，则计算 max_degree 并重新加载带 transform 的数据集
    if tmp.num_node_features == 0:
        max_deg = 0
        for data in tmp:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_deg = max(max_deg, int(deg.max()))
        dataset = TUDataset(root + '/TUDataset', dataset_name, transform=OneHotDegree(max_deg))
    else:
        dataset = tmp

    print(f"COLORS-3数据集原始总大小: {len(dataset)}")
    print(
        f"节点特征维度: {dataset[0].x.shape[1] if hasattr(dataset[0], 'x') and dataset[0].x is not None else '无节点特征'}")
    print(f"数据缩放因子: {scale_factor}")

    # 3. 收集所有标签并确保为整数类型
    labels = []
    for data in dataset:
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        labels.append(int(label))  # 确保为整数
    labels = torch.tensor(labels, dtype=torch.long)  # 明确指定为long类型

    # 4. 获取所有唯一类别并创建标签映射
    all_classes = sorted(labels.unique().tolist())
    known_classes = [c for c in all_classes if c not in unknown_classes]

    # 创建从原始标签到新标签的映射
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")

    # 5. 数据缩放逻辑
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放COLORS-3数据集 (缩放因子: {scale_factor}) ===")

        # 按类别分组所有样本
        all_class_indices = {}
        for idx in range(len(dataset)):
            label = labels[idx].item()
            if label not in all_class_indices:
                all_class_indices[label] = []
            all_class_indices[label].append(idx)

        # 对每个类别等比例采样
        scaled_indices = []
        for class_label in all_classes:
            if class_label in all_class_indices:
                class_indices = all_class_indices[class_label]
                original_count = len(class_indices)
                scaled_count = max(1, int(original_count * scale_factor))  # 至少保留1个样本

                # 随机采样
                random.shuffle(class_indices)
                sampled_indices = class_indices[:scaled_count]
                scaled_indices.extend(sampled_indices)

                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        # 重新构建缩放后的数据集和标签
        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices], dtype=torch.long)

        # 更新数据集和标签
        dataset = scaled_dataset
        labels = scaled_labels

        print(f"缩放后数据集大小: {len(dataset)}")

    # 6. 分离已知和未知样本
    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 7. 分层采样 - 按类别分别划分train/test
    # 按类别分组已知样本的索引
    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:  # 只处理已知类别
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== COLORS-3分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:  # 添加检查，防止缩放后某些类别消失
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train

            # 随机打乱该类别的样本
            random.shuffle(indices)

            # 分配到训练集和测试集
            class_train = indices[:n_train]
            class_test = indices[n_train:]

            train_idx.extend(class_train)
            test_idx.extend(class_test)

            print(f"类别 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    # 8. 打乱最终的训练和测试索引
    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 9. 调整未知样本数量以匹配测试集大小
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        # 如果未知样本太多，随机采样
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        # 如果未知样本太少，保持原有数量但给出提示
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 10. 创建重新映射标签的数据集包装器
    class RelabeledCOLORS3Subset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]

            # 重新映射标签，确保类型正确
            if self.is_unknown:
                # 所有未知类别统一映射为 num_known
                data.y = torch.tensor(self.num_known, dtype=torch.long, device=data.y.device)
            else:
                # 已知类别重新映射
                original_label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
                original_label = int(original_label)  # 确保为整数

                if original_label in self.label_mapping:
                    new_label = self.label_mapping[original_label]
                    data.y = torch.tensor(new_label, dtype=torch.long, device=data.y.device)

            return data

    # 11. 构造返回的数据集
    num_known = len(known_classes)

    train_known = RelabeledCOLORS3Subset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledCOLORS3Subset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledCOLORS3Subset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    # 12. 验证分层采样效果
    print(f"\n=== 验证COLORS-3训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:  # 添加安全检查
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_reddit_multi_5k_dataset(
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        max_degree: int = 20  # 固定最大度数，特征维度为21
):
    """
    专门为REDDIT-MULTI-5K数据集设计的简化分割函数

    Args:
        unknown_classes: 未知类别列表
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        scale_factor: 数据缩放因子（0-1之间）
        max_degree: 最大度数限制，控制特征维度
    """
    import random
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import degree, one_hot

    dataset_name = 'REDDIT-MULTI-5K'

    print(f"=== 加载REDDIT-MULTI-5K数据集 ===")
    print(f"最大度数限制: {max_degree} (特征维度: {max_degree + 1})")
    print(f"数据缩放因子: {scale_factor}")

    # 1. 创建自定义变换：截断度数 + OneHot编码
    class SimpleClampedDegree:
        def __init__(self, max_deg):
            self.max_deg = max_deg

        def __call__(self, data):
            # 计算度数
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            # 截断过大的度数
            deg = torch.clamp(deg, 0, self.max_deg)
            # OneHot编码
            data.x = one_hot(deg, num_classes=self.max_deg + 1).float()
            return data

    # 2. 加载数据集
    dataset = TUDataset(
        root + '/TUDataset',
        dataset_name,
        transform=SimpleClampedDegree(max_degree)
    )

    print(f"原始数据集大小: {len(dataset)}")
    print(f"节点特征维度: {max_degree + 1}")

    # 3. 收集所有标签
    labels = []
    for data in dataset:
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        labels.append(int(label))
    labels = torch.tensor(labels, dtype=torch.long)

    # 4. 确定类别信息
    all_classes = [0, 1, 2, 3, 4]  # REDDIT-MULTI-5K有5个类别
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")

    # 5. 数据缩放
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放数据集 (缩放因子: {scale_factor}) ===")

        # 按类别分组
        class_indices = {}
        for idx, label in enumerate(labels):
            label = label.item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # 对每个类别等比例采样
        scaled_indices = []
        for class_label in all_classes:
            if class_label in class_indices:
                indices = class_indices[class_label]
                original_count = len(indices)
                scaled_count = max(1, int(original_count * scale_factor))

                random.shuffle(indices)
                sampled_indices = indices[:scaled_count]
                scaled_indices.extend(sampled_indices)

                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        # 重建数据集
        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        dataset = scaled_dataset
        labels = scaled_labels
        print(f"缩放后数据集大小: {len(dataset)}")

    # 6. 分离已知和未知样本
    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 7. 按类别分层采样
    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== REDDIT-MULTI-5K 分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train

            random.shuffle(indices)
            class_train = indices[:n_train]
            class_test = indices[n_train:]

            train_idx.extend(class_train)
            test_idx.extend(class_test)

            print(f"类别 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    # 8. 打乱索引
    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 9. 调整未知样本数量
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 10. 创建数据集包装器
    class SimpleRedditSubset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]

            if self.is_unknown:
                # 未知类别标记为num_known
                data.y = torch.tensor(self.num_known, dtype=torch.long, device=data.y.device)
            else:
                # 已知类别重新映射
                original_label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=torch.long, device=data.y.device)

            return data

    # 11. 构造返回的数据集
    num_known = len(known_classes)

    train_known = SimpleRedditSubset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = SimpleRedditSubset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = SimpleRedditSubset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    # 12. 验证结果
    print(f"\n=== 验证REDDIT-MULTI-5K训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown

def load_dataset_unified(dataset_name, unknown_classes, ratio, root, seed, **kwargs):
    """
    统一的数据集加载函数，支持多种数据集类型

    Args:
        dataset_name: 数据集名称
        unknown_classes: 未知类别列表
        ratio: 训练/测试分割比例
        root: 数据根目录
        seed: 随机种子
        **kwargs: 额外参数
            - scale_factor: 数据缩放因子 (所有数据集)
            - num_points: ModelNet点云采样点数 (默认1024)
            - k_neighbors: ModelNet k-NN邻居数 (默认6)
    """
    scale_factor = kwargs.get('scale_factor', 1.0)

    if dataset_name == 'REDDIT-MULTI-5K':
        max_degree = kwargs.get('max_degree', 50)  # 默认最大度数20
        return split_reddit_multi_5k_dataset(unknown_classes, ratio, root, seed, scale_factor, max_degree)

    if dataset_name == 'MNIST':
        return split_mnist_dataset(unknown_classes, ratio, root, seed, scale_factor)
    elif dataset_name == 'COLORS-3':
        return split_colors3_dataset(unknown_classes, ratio, root, seed, scale_factor)
    elif dataset_name in ['CLUSTER', 'CSL', 'CIFAR10']:  # GNNBenchmarkDataset非二分类数据集
        return split_gnn_benchmark_dataset(dataset_name, unknown_classes, ratio, root, seed, scale_factor)

    elif dataset_name in ['10', '40']:  # ModelNet数据集
        num_points = kwargs.get('num_points', 1024)
        k_neighbors = kwargs.get('k_neighbors', 6)
        return split_modelnet_dataset(dataset_name, unknown_classes, ratio, root, seed,
                                      scale_factor, num_points, k_neighbors)

    else:

        # 修复：现在TU数据集也支持scale_factor了！
        return split_dataset(dataset_name, unknown_classes, ratio, root, seed, scale_factor)


def split_gnn_benchmark_dataset(
        dataset_name: str,
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0
):
    """
    专门为GNNBenchmarkDataset设计的分割函数
    只支持非二分类数据集: CLUSTER (6类), CSL (10类), MNIST (10类), CIFAR10 (10类)
    """
    from torch_geometric.datasets import GNNBenchmarkDataset
    import torch
    import random

    # 检查是否为非二分类数据集
    non_binary_datasets = {
        'CLUSTER': 6,  # 6类
        'CSL': 10,  # 10类
        'MNIST': 10,  # 10类
        'CIFAR10': 10  # 10类
    }

    binary_datasets = ['PATTERN', 'TSP']  # 2类数据集

    if dataset_name in binary_datasets:
        raise ValueError(f"数据集 {dataset_name} 是二分类数据集，不支持。"
                         f"请使用非二分类数据集: {list(non_binary_datasets.keys())}")

    if dataset_name not in non_binary_datasets:
        raise ValueError(f"不支持的数据集 {dataset_name}。"
                         f"支持的数据集: {list(non_binary_datasets.keys())}")

    print(f"=== 加载GNNBenchmarkDataset: {dataset_name} ===")

    # 加载训练、验证、测试集
    train_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='train')
    val_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='val')
    test_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='test')

    # 合并所有数据集
    all_dataset = list(train_dataset) + list(val_dataset) + list(test_dataset)
    num_classes = non_binary_datasets[dataset_name]

    print(f"原始数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    print(f"合并后总大小: {len(all_dataset)}")
    print(
        f"节点特征维度: {all_dataset[0].x.shape[1] if hasattr(all_dataset[0], 'x') and all_dataset[0].x is not None else '无节点特征'}")
    print(f"总类别数: {num_classes}")
    print(f"数据缩放因子: {scale_factor}")

    # 收集所有标签
    labels = torch.tensor([data.y.item() if data.y.numel() == 1 else data.y[0].item() for data in all_dataset])

    # 获取所有唯一类别并创建标签映射
    all_classes = list(range(num_classes))
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")

    # 数据缩放
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放数据集 (缩放因子: {scale_factor}) ===")
        all_class_indices = {}
        for idx in range(len(all_dataset)):
            label = labels[idx].item()
            if label not in all_class_indices:
                all_class_indices[label] = []
            all_class_indices[label].append(idx)

        scaled_indices = []
        for class_label in all_classes:
            if class_label in all_class_indices:
                class_indices = all_class_indices[class_label]
                original_count = len(class_indices)
                scaled_count = max(1, int(original_count * scale_factor))
                random.shuffle(class_indices)
                sampled_indices = class_indices[:scaled_count]
                scaled_indices.extend(sampled_indices)
                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])
        all_dataset = scaled_dataset
        labels = scaled_labels
        print(f"缩放后数据集大小: {len(all_dataset)}")

    # 分离已知和未知样本
    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 分层采样
    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== {dataset_name} 分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train
            random.shuffle(indices)
            class_train = indices[:n_train]
            class_test = indices[n_train:]
            train_idx.extend(class_train)
            test_idx.extend(class_test)
            print(f"类别 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 调整未知样本数量
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 创建重新映射标签的数据集包装器
    class RelabeledGNNBenchmarkSubset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]
            if self.is_unknown:
                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
            else:
                original_label = data.y.flatten()[0].item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    num_known = len(known_classes)
    train_known = RelabeledGNNBenchmarkSubset(all_dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledGNNBenchmarkSubset(all_dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledGNNBenchmarkSubset(all_dataset, unknown_idx, label_mapping, is_unknown=True,
                                              num_known=num_known)

    print(f"\n=== 验证{dataset_name}训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1
    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_modelnet_dataset(
        dataset_name: str,  # "10" 或 "40"
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        num_points: int = 1024,
        k_neighbors: int = 6
):
    """
    专门为ModelNet数据集设计的分割函数
    将3D网格转换为点云图进行处理
    """
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, KNNGraph, Compose
    import torch
    import random

    if dataset_name not in ["10", "40"]:
        raise ValueError(f"ModelNet数据集名称必须是 '10' 或 '40'，得到: {dataset_name}")

    print(f"=== 加载ModelNet{dataset_name}数据集 ===")

    # 添加自定义变换类：将3D坐标作为节点特征
    class AddPositionAsFeature:
        """将点的3D坐标作为节点特征用于图分类"""
        def __call__(self, data):
            data.x = data.pos  # pos是[N, 3]的3D坐标，作为[N, 3]的节点特征
            return data

    # 定义转换：网格 -> 点云 -> k-NN图 -> 添加节点特征
    transform = Compose([
        SamplePoints(num_points, remove_faces=False, include_normals=False),
        KNNGraph(k=k_neighbors),
        AddPositionAsFeature()  # 将3D坐标作为节点特征
    ])

    # 加载训练和测试集
    train_dataset = ModelNet(
        root=root + f'/ModelNet{dataset_name}',
        name=dataset_name,
        train=True,
        transform=transform
    )

    test_dataset = ModelNet(
        root=root + f'/ModelNet{dataset_name}',
        name=dataset_name,
        train=False,
        transform=transform
    )

    all_dataset = list(train_dataset) + list(test_dataset)
    num_classes = 10 if dataset_name == "10" else 40

    print(f"原始数据集大小: 训练={len(train_dataset)}, 测试={len(test_dataset)}")
    print(f"合并后总大小: {len(all_dataset)}")
    print(f"点云点数: {num_points}")
    print(f"k-NN邻居数: {k_neighbors}")
    print(f"节点特征维度: 3 (x, y, z坐标)")
    print(f"总类别数: {num_classes}")
    print(f"数据缩放因子: {scale_factor}")

    # 收集所有标签
    labels = torch.tensor([int(data.y) for data in all_dataset])

    # 获取所有唯一类别并创建标签映射
    all_classes = list(range(num_classes))
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"原始类别: {all_classes}")
    print(f"未知类别: {unknown_classes}")
    print(f"已知类别: {known_classes}")
    print(f"标签映射: {label_mapping}")

    # 数据缩放
    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== 等比例缩放数据集 (缩放因子: {scale_factor}) ===")
        all_class_indices = {}
        for idx in range(len(all_dataset)):
            label = labels[idx].item()
            if label not in all_class_indices:
                all_class_indices[label] = []
            all_class_indices[label].append(idx)

        scaled_indices = []
        for class_label in all_classes:
            if class_label in all_class_indices:
                class_indices = all_class_indices[class_label]
                original_count = len(class_indices)
                scaled_count = max(1, int(original_count * scale_factor))
                random.shuffle(class_indices)
                sampled_indices = class_indices[:scaled_count]
                scaled_indices.extend(sampled_indices)
                print(f"类别 {class_label}: {original_count} -> {scaled_count} 样本")

        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])
        all_dataset = scaled_dataset
        labels = scaled_labels
        print(f"缩放后数据集大小: {len(all_dataset)}")

    # 分离已知和未知样本
    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    # 分层采样
    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== ModelNet{dataset_name} 分层采样详情 ===")
    for label in known_classes:
        if label in class_indices:
            indices = class_indices[label]
            n_samples = len(indices)
            n_train = int(n_samples * ratio)
            n_test = n_samples - n_train
            random.shuffle(indices)
            class_train = indices[:n_train]
            class_test = indices[n_train:]
            train_idx.extend(class_train)
            test_idx.extend(class_test)
            print(f"类别 {label}: 总数={n_samples}, 训练={n_train}, 测试={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    # 调整未知样本数量
    test_known_size = len(test_idx)
    print(f"测试集（已知类别）样本数: {test_known_size}")
    print(f"原始未知样本数: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"从未知样本中随机采样 {test_known_size} 个样本")
    elif len(unknown_idx) < test_known_size:
        print(f"警告：未知样本数量({len(unknown_idx)})少于测试集样本数量({test_known_size})")
        print(f"保持所有 {len(unknown_idx)} 个未知样本")
    else:
        print(f"未知样本数量已经与测试集匹配: {len(unknown_idx)}")

    print(f"总训练样本: {len(train_idx)}")
    print(f"总测试样本: {len(test_idx)}")
    print(f"最终未知样本: {len(unknown_idx)}")

    # 创建重新映射标签的数据集包装器
    class RelabeledModelNetSubset:
        def __init__(self, dataset, indices, label_mapping, is_unknown=False, num_known=None):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping
            self.is_unknown = is_unknown
            self.num_known = num_known

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data = self.dataset[self.indices[idx]]
            if self.is_unknown:
                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
            else:
                original_label = data.y.item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    num_known = len(known_classes)
    train_known = RelabeledModelNetSubset(all_dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledModelNetSubset(all_dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledModelNetSubset(all_dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    print(f"\n=== 验证ModelNet{dataset_name}训练集类别分布 ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1
    print(f"训练集重映射标签分布: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown



import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, MLP

class GINSWDSingleModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            initial_atoms: torch.Tensor,  # [num_known, K, D]
            n_projections: int = 100,
            seed: int = 0,
            mlp_layers: int = 3,
            use_bn: bool = True,
            train_eps: bool = False,
            swd_method: str = "mmd",
            gamma = None
    ):
        super().__init__()

        # 1. GIN卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            mlp = MLP(inc, hidden_channels, hidden_channels, mlp_layers, use_bn)
            self.convs.append(GINConv(mlp, train_eps=train_eps))

        # 2. Atom 参数
        self.atoms = nn.Parameter(initial_atoms)  # [C, K, D]

        # 3. 选择方法
        self.n_projections = n_projections
        self.seed = seed
        self.swd_method = swd_method

        if swd_method == "geomloss":
            from geomloss import SamplesLoss
            self.swd_loss = SamplesLoss(
                loss="sinkhorn",
                p=1,
                blur=0.01,
                scaling=0.9,
                backend="tensorized"
            )
        elif swd_method == "pot":
            pass
        elif swd_method == "mmd":
            # 用RBF核的MMD，无需额外库
            C, K, D = initial_atoms.shape
            # 设置默认带宽为 1/D
            if gamma is None:
                self.gamma = 1.0 / D
            else:
                self.gamma = gamma
        else:
            # manual SWD
            C, K, D = initial_atoms.shape
            thetas = torch.randn(n_projections, D, device=initial_atoms.device)
            thetas = thetas / thetas.norm(dim=1, keepdim=True)
            self.register_buffer('thetas', thetas)

    def forward(self, data_or_x, batch=None, edge_index=None):
        if batch is None:
            data = data_or_x
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = conv(x, edge_index)
        else:
            x = data_or_x
        return self._swd(x, batch)

    def extract_node_features(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

    def _swd(self, x_mat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        device = x_mat.device
        C, K, D = self.atoms.shape
        bs = int(batch.max()) + 1

        if self.swd_method == "geomloss":
            swd_distances = torch.zeros(bs, C, device=device)
            for i in range(bs):
                Xi = x_mat[batch == i]
                for j in range(C):
                    Aj = self.atoms[j]
                    swd_distances[i, j] = self.swd_loss(Xi, Aj)
            return swd_distances

        elif self.swd_method == "pot":
            import ot
            import numpy as np
            swd_distances = torch.zeros(bs, C, device=device)
            for i in range(bs):
                Xi = x_mat[batch == i].detach().cpu().numpy()
                for j in range(C):
                    Aj = self.atoms[j].detach().cpu().numpy()
                    a = np.ones(Xi.shape[0]) / Xi.shape[0]
                    b = np.ones(Aj.shape[0]) / Aj.shape[0]
                    swd_val = ot.sliced_wasserstein_distance(
                        Xi, Aj, a, b,
                        n_projections=self.n_projections,
                        seed=self.seed
                    )
                    swd_distances[i, j] = torch.tensor(swd_val, device=device)
            return swd_distances

        elif self.swd_method == "mmd":
            # MMD: E[k(x,x)] + E[k(y,y)] - 2 E[k(x,y)]
            mmd_distances = torch.zeros(bs, C, device=device)
            for i in range(bs):
                Xi = x_mat[batch == i]
                for j in range(C):
                    Aj = self.atoms[j]
                    mmd_distances[i, j] = self._mmd(Xi, Aj)
            return mmd_distances

        else:
            # manual SWD
            torch.manual_seed(self.seed)
            swd_values = []
            C, K, D = self.atoms.shape
            for i in range(bs):
                Xi = x_mat[batch == i]
                row = []
                for j in range(C):
                    Aj = self.atoms[j]
                    Xi_proj = Xi @ self.thetas.t()
                    Aj_proj = Aj @ self.thetas.t()
                    Xi_ps, _ = torch.sort(Xi_proj, dim=0)
                    Aj_ps, _ = torch.sort(Aj_proj, dim=0)
                    if Xi_ps.size(0) != K:
                        Xi_ps = self._differentiable_interpolate(Xi_ps, K)
                    w = torch.mean(torch.abs(Xi_ps - Aj_ps), dim=0)
                    row.append(torch.mean(w))
                swd_values.append(torch.stack(row))
            return torch.stack(swd_values)

    def _mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # RBF kernel MMD
        gamma = self.gamma
        xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(2)
        k_xx = torch.exp(-gamma * xx).mean()
        yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        k_yy = torch.exp(-gamma * yy).mean()
        xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        k_xy = torch.exp(-gamma * xy).mean()
        return k_xx + k_yy - 2 * k_xy

    def _differentiable_interpolate(self, x, target_size):
        current_size = x.size(0)
        if current_size == target_size:
            return x
        indices = torch.linspace(0, current_size - 1, target_size, device=x.device)
        floor_idx = indices.floor().long().clamp(0, current_size-1)
        ceil_idx = indices.ceil().long().clamp(0, current_size-1)
        weights = indices - floor_idx.float()
        return (1 - weights.unsqueeze(1)) * x[floor_idx] + weights.unsqueeze(1) * x[ceil_idx]

    def atom_distances(self) -> torch.Tensor:
        C, K, D = self.atoms.shape
        device = self.atoms.device
        distances = torch.zeros(C, C, device=device)

        if self.swd_method == "geomloss":
            for i in range(C):
                for j in range(C):
                    if i != j:
                        distances[i, j] = self.swd_loss(self.atoms[i], self.atoms[j])

        elif self.swd_method == "pot":
            import ot
            import numpy as np
            for i in range(C):
                for j in range(C):
                    if i != j:
                        Ai = self.atoms[i].detach().cpu().numpy()
                        Aj = self.atoms[j].detach().cpu().numpy()
                        a = np.ones(K) / K
                        b = np.ones(K) / K
                        swd_val = ot.sliced_wasserstein_distance(
                            Ai, Aj, a, b,
                            n_projections=self.n_projections,
                            seed=self.seed
                        )
                        distances[i, j] = torch.tensor(swd_val, device=device)

        elif self.swd_method == "mmd":
            for i in range(C):
                for j in range(C):
                    if i != j:
                        distances[i, j] = self._mmd(self.atoms[i], self.atoms[j])

        else:
            torch.manual_seed(self.seed)
            for i in range(C):
                for j in range(C):
                    if i != j:
                        Ai = self.atoms[i]
                        Aj = self.atoms[j]
                        Ai_p = Ai @ self.thetas.t()
                        Aj_p = Aj @ self.thetas.t()
                        Ai_ps, _ = torch.sort(Ai_p, dim=0)
                        Aj_ps, _ = torch.sort(Aj_p, dim=0)
                        w = torch.mean(torch.abs(Ai_ps - Aj_ps), dim=0)
                        distances[i, j] = torch.mean(w)

        return distances



class MLP(torch.nn.Module):
    """MLP结构"""

    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, use_bn=True):
        super().__init__()
        from torch.nn import ModuleList, Linear, BatchNorm1d

        self.num_layers = num_layers
        self.use_bn = use_bn
        self.linears = ModuleList()
        self.bns = ModuleList() if use_bn else None

        # 第一层
        self.linears.append(Linear(input_channels, hidden_channels))
        if use_bn:
            self.bns.append(BatchNorm1d(hidden_channels))

        # 中间层
        for layer in range(num_layers - 2):
            self.linears.append(Linear(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(BatchNorm1d(hidden_channels))

        # 输出层
        if num_layers > 1:
            self.linears.append(Linear(hidden_channels, output_channels))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)

        # 最后一层不加激活
        if self.num_layers > 1:
            x = self.linears[-1](x)
        return x

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        if self.use_bn:
            for layer in self.bns:
                layer.reset_parameters()


import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU


def init_atoms_from_dataset(
        dataset,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,  # 只包含已知类的数量
        num_atom_supp: int,
        device: torch.device
) -> torch.Tensor:
    """
    原子初始化，使用真正的GIN提取节点嵌入 - 只初始化已知类
    """
    # 1. 构造一组 GINConv 层（与模型里一致）
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

    # 2. 遍历 dataset，提取每张图的节点嵌入、节点数、标签
    graph_embs, node_counts, labels = [], [], []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            x, edge_index = data.x, data.edge_index
            # 使用真正的GIN卷积
            for conv in convs:
                x = conv(x, edge_index)
            graph_embs.append(x.cpu())  # [n_i, D]
            node_counts.append(x.size(0))
            labels.append(int(data.y.item()))

    # 3. 对每个已知类别做原子初始化
    atoms = []
    for c in range(num_classes):  # 只处理已知类
        idxs = [i for i, lbl in enumerate(labels) if lbl == c]
        # 3.1 直取节点数正好的那张图
        sel = None
        for i in idxs:
            if node_counts[i] == num_atom_supp:
                sel = graph_embs[i]
                break
        if sel is not None:
            atoms.append(sel)
        else:
            # 聚所有节点做 KMeans
            all_nodes = torch.cat([graph_embs[i] for i in idxs], dim=0)  # [M, D]
            km = KMeans(n_clusters=num_atom_supp, random_state=0).fit(all_nodes.numpy())
            centers = torch.from_numpy(km.cluster_centers_)  # [K, D]
            atoms.append(centers)

    # 4. Stack → [num_classes, num_atom_supp, D] - 只包含已知类
    return torch.stack(atoms, dim=0)


import torch


def sample_noise_features(model, data_batch, ratio=0.3, distance_factor=5.0, noise_sigma=3.0):
    """
    简单版本：随机采样 + 加噪生成负样本
    """
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    # 1. 根据ratio选择样本
    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    # 2. 提取节点特征
    all_node_features = model.extract_node_features(data_batch)

    neg_features_list = []
    neg_batch_list = []

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        graph_features = all_node_features[mask]  # 原始图的节点特征

        # 3. 简单处理：直接加噪声
        neg_features = graph_features.clone()

        # 加高斯噪声
        noise = torch.randn_like(neg_features) * noise_sigma
        neg_features += noise

        graph_batch = torch.full((graph_features.size(0),), new_graph_id,
                                 dtype=torch.long, device=device)

        neg_features_list.append(neg_features)
        neg_batch_list.append(graph_batch)

    if len(neg_features_list) > 0:
        x_neg = torch.cat(neg_features_list, dim=0)
        neg_batch = torch.cat(neg_batch_list, dim=0)
    else:
        hidden_dim = all_node_features.size(1)
        x_neg = torch.empty(0, hidden_dim, device=device)
        neg_batch = torch.empty(0, dtype=torch.long, device=device)

    return x_neg, neg_batch

def sample_neg_features_directed_perturbation(model, data_batch, ratio=0.3, multiplier=1.2,
                                              max_attempts=30, distance_step=0.5):
    """
    基于有方向扰动的负样本生成：
    1. 计算正样本到其对应atom的MMD距离，确定threshold
    2. 选择部分正样本，朝着远离atoms的方向扰动

    Args:
        model: 模型对象
        data_batch: 数据批次
        ratio: 选择样本的比例 (0-1)
        multiplier: threshold倍数（相对于正样本baseline距离）
        max_attempts: 最大尝试次数
        distance_step: 每次尝试增加的距离步长

    Returns:
        x_neg: 负样本节点特征
        neg_batch: 负样本批次标识
    """
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    # 1. 提取节点特征和atoms
    all_node_features = model.extract_node_features(data_batch)
    atoms = model.atoms  # [C, S, D]
    num_classes, num_atom_supp, hidden_dim = atoms.shape

    # 使用和模型一致的gamma参数
    gamma = model.gamma  # 1.0 / D


    # 2. 计算所有正样本到其对应atom的MMD距离，确定baseline
    positive_distances = []

    for i in range(batch_size):
        mask = data_batch.batch == i
        graph_features = all_node_features[mask]  # [N, D]
        graph_label = data_batch.y[i].item()  # 标签

        # 计算该正样本到其对应atom的MMD距离
        target_atom = atoms[graph_label]  # [S, D]
        mmd_dist = compute_mmd_your_way(graph_features, target_atom, gamma)
        positive_distances.append(mmd_dist)

    # 3. 确定threshold
    baseline_distance = np.mean(positive_distances)
    threshold = baseline_distance * multiplier



    # 4. 预计算全局参考信息
    atoms_center = atoms.mean(dim=(0, 1))  # [D] - 所有atoms的中心

    # 计算特征的典型尺度（用于控制扰动强度）
    all_features_norm = torch.norm(all_node_features, dim=1).mean().item()
    feature_std = all_node_features.std().item()


    # 5. 根据ratio选择要扰动的正样本
    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    neg_features_list = []
    neg_batch_list = []

    success_count = 0


    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        original_features = all_node_features[mask]  # [N, D] 原始正样本特征
        num_nodes = original_features.size(0)

        # 6. 计算图的中心和远离atoms的方向
        graph_center = original_features.mean(dim=0)  # [D]

        # 计算从atoms中心指向图中心的方向（我们要沿着这个方向或反方向移动）
        direction_to_graph = graph_center - atoms_center  # [D]
        direction_norm = direction_to_graph.norm()

        if direction_norm > 1e-8:
            # 标准化方向向量
            push_direction = direction_to_graph / direction_norm  # [D]
        else:
            # 如果图中心和atoms中心重合，随机选择一个方向
            push_direction = torch.randn_like(graph_center)
            push_direction = push_direction / push_direction.norm()

        # 7. 逐步增加推离距离，直到满足threshold要求
        found_valid = False

        for attempt in range(max_attempts):
            # 计算当前尝试的推离距离
            push_distance = distance_step * (attempt + 1)

            # 沿着远离atoms的方向推离
            directed_push = push_direction.unsqueeze(0).repeat(num_nodes, 1) * push_distance

            # 添加小幅随机噪声增加多样性（控制在特征尺度内）
            random_noise = torch.randn_like(original_features) * feature_std * 0.1

            # 生成扰动后的特征
            perturbed_features = original_features + directed_push + random_noise

            # 8. 检查扰动后的特征与所有atoms的MMD距离
            min_mmd_to_atoms = float('inf')

            for class_idx in range(num_classes):
                current_atom = atoms[class_idx]  # [S, D]
                mmd_dist = compute_mmd_your_way(perturbed_features, current_atom, gamma)
                min_mmd_to_atoms = min(min_mmd_to_atoms, mmd_dist)

            # 9. 检查是否满足threshold要求
            if min_mmd_to_atoms >= threshold:
                # 找到满足条件的负样本
                neg_features_list.append(perturbed_features)

                graph_batch = torch.full((num_nodes,), new_graph_id,
                                         dtype=torch.long, device=device)
                neg_batch_list.append(graph_batch)

                found_valid = True
                success_count += 1


                break

        if not found_valid:
            # 如果达到最大尝试次数仍未成功，使用最大推离距离的结果
            max_push_distance = distance_step * max_attempts
            directed_push = push_direction.unsqueeze(0).repeat(num_nodes, 1) * max_push_distance
            random_noise = torch.randn_like(original_features) * feature_std * 0.1
            perturbed_features = original_features + directed_push + random_noise

            # 重新计算距离用于记录
            min_mmd_to_atoms = float('inf')
            for class_idx in range(num_classes):
                current_atom = atoms[class_idx]
                mmd_dist = compute_mmd_your_way(perturbed_features, current_atom, gamma)
                min_mmd_to_atoms = min(min_mmd_to_atoms, mmd_dist)

            neg_features_list.append(perturbed_features)

            graph_batch = torch.full((num_nodes,), new_graph_id,
                                     dtype=torch.long, device=device)
            neg_batch_list.append(graph_batch)


    # 10. 合并所有负样本
    if len(neg_features_list) > 0:
        x_neg = torch.cat(neg_features_list, dim=0)
        neg_batch = torch.cat(neg_batch_list, dim=0)
    else:
        x_neg = torch.empty(0, hidden_dim, device=device)
        neg_batch = torch.empty(0, dtype=torch.long, device=device)

    success_rate = success_count / len(selected_indices) if selected_indices else 0

    return x_neg, neg_batch


def compute_mmd_your_way(x: torch.Tensor, y: torch.Tensor, gamma: float) -> float:
    """
    完全按照你的模型实现MMD计算

    Args:
        x: 第一组样本 [N1, D]
        y: 第二组样本 [N2, D]
        gamma: RBF核参数 (通常是 1.0 / D)

    Returns:
        mmd_distance: MMD距离值
    """
    # 完全复制你的_mmd函数实现
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(2)
    k_xx = torch.exp(-gamma * xx).mean()

    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    k_yy = torch.exp(-gamma * yy).mean()

    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    k_xy = torch.exp(-gamma * xy).mean()

    mmd_result = k_xx + k_yy - 2 * k_xy

    return mmd_result.item()

def sample_neg_features_improved(model, data_batch, selected_indices, sigma=2.0, p_drop=0.3):
    """
    改进的负样本生成：更大的噪声，更激进的扰动
    """
    device = data_batch.x.device

    if torch.is_tensor(selected_indices):
        selected_indices = selected_indices.tolist()

    if len(selected_indices) == 0:
        hidden_dim = model.atoms.shape[-1]
        return (torch.empty(0, hidden_dim, device=device),
                torch.empty(0, dtype=torch.long, device=device))

    all_node_features = model.extract_node_features(data_batch)

    neg_features_list = []
    neg_batch_list = []

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        graph_features = all_node_features[mask]

        # 1. 更大的高斯噪声
        noise = torch.randn_like(graph_features) * sigma  # sigma=2.0, 更大

        # 2. 随机反转一些特征的符号
        flip_mask = torch.rand_like(graph_features) < 0.3
        noise[flip_mask] *= -1

        # 3. 添加随机偏移，推向atoms的反方向
        atoms_center = model.atoms.mean(dim=(0, 1))  # [D]
        push_direction = torch.randn_like(atoms_center)
        push_direction = push_direction / push_direction.norm()

        neg_features = graph_features + noise + push_direction.unsqueeze(0) * 3.0

        # 4. 更激进的dropout
        dropout_mask = torch.rand(graph_features.size(0), device=device) < p_drop
        neg_features[dropout_mask] = torch.randn_like(neg_features[dropout_mask]) * sigma

        graph_batch = torch.full((graph_features.size(0),), new_graph_id,
                                 dtype=torch.long, device=device)

        neg_features_list.append(neg_features)
        neg_batch_list.append(graph_batch)

    if len(neg_features_list) > 0:
        x_neg = torch.cat(neg_features_list, dim=0)
        neg_batch = torch.cat(neg_batch_list, dim=0)
    else:
        hidden_dim = all_node_features.size(1)
        x_neg = torch.empty(0, hidden_dim, device=device)
        neg_batch = torch.empty(0, dtype=torch.long, device=device)

    return x_neg, neg_batch


def sample_anti_atoms_negatives(model, num_neg):
    """在atoms的"反方向"采样负样本"""
    atoms = model.atoms  # [C, K, D]
    C, K, D = atoms.shape
    device = atoms.device

    # 计算atoms的中心
    atoms_center = atoms.mean(dim=(0, 1))  # [D]

    neg_features_list = []
    neg_batch_list = []

    for i in range(num_neg):
        # 在远离atoms中心的方向采样
        random_direction = torch.randn(D, device=device)
        random_direction = random_direction / random_direction.norm()

        # 距离atoms中心较远的点
        distance = torch.rand(1, device=device) * 5 + 2  # [2, 7]的距离
        neg_center = atoms_center + random_direction * distance

        # 在这个点周围生成节点
        neg_feat = neg_center.unsqueeze(0).repeat(10, 1)
        neg_feat += torch.randn_like(neg_feat) * 0.5

        neg_features_list.append(neg_feat)
        neg_batch_list.append(torch.full((10,), i, dtype=torch.long, device=device))

    return torch.cat(neg_features_list), torch.cat(neg_batch_list)


def sample_neg_features_atoms(model, data_batch, ratio=0.3, distance_factor=5.0, noise_sigma=3.0):
    """
    基于比例的负样本生成：从数据批次中选择一定比例的样本，生成远离atoms的负样本

    Args:
        model: 模型对象
        data_batch: 数据批次
        ratio: 选择样本的比例 (0-1)
        distance_factor: 距离atoms的倍数
        noise_sigma: 噪声强度

    Returns:
        x_neg: 负样本节点特征
        neg_batch: 负样本批次标识
    """
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    # 1. 根据ratio选择样本
    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    # 2. 提取节点特征
    all_node_features = model.extract_node_features(data_batch)

    # 3. 计算atoms中心作为参考点
    atoms_center = model.atoms.mean(dim=(0, 1))  # [D]

    neg_features_list = []
    neg_batch_list = []

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        graph_features = all_node_features[mask]

        # 4. 生成远离atoms的负样本
        # 计算从atoms中心到图特征的方向
        graph_center = graph_features.mean(dim=0)
        direction = graph_center - atoms_center
        direction = direction / (direction.norm() + 1e-8)

        # 在相反方向上生成负样本
        neg_center = atoms_center - direction * distance_factor

        # 在负样本中心周围添加噪声
        neg_features = neg_center.unsqueeze(0).repeat(graph_features.size(0), 1)
        neg_features += torch.randn_like(neg_features) * noise_sigma

        # 额外扰动：添加随机偏移
        random_offset = torch.randn_like(neg_features) * 2.0
        neg_features += random_offset

        graph_batch = torch.full((graph_features.size(0),), new_graph_id,
                                 dtype=torch.long, device=device)

        neg_features_list.append(neg_features)
        neg_batch_list.append(graph_batch)

    if len(neg_features_list) > 0:
        x_neg = torch.cat(neg_features_list, dim=0)
        neg_batch = torch.cat(neg_batch_list, dim=0)
    else:
        hidden_dim = all_node_features.size(1)
        x_neg = torch.empty(0, hidden_dim, device=device)
        neg_batch = torch.empty(0, dtype=torch.long, device=device)

    return x_neg, neg_batch

def find_otsu_threshold(distances):
    """
    使用Otsu方法自动找到最优阈值
    """
    distances = np.array(distances)

    # 将连续值离散化为直方图
    hist, bin_edges = np.histogram(distances, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu算法
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

        # 类间方差
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance_between > current_max:
            current_max = variance_between
            threshold_idx = i

    # 转换回原始距离值
    threshold_value = bin_centers[min(threshold_idx, len(bin_centers) - 1)]

    print(f"🎯 Otsu自动检测阈值: {threshold_value:.4f}")
    return threshold_value


import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter, find_peaks


import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter, find_peaks

import numpy as np


def find_gradient_gap_threshold(distances):
    """
    基于密度梯度的Gap检测 - 找到真正的第一个大波谷
    策略：找到第一个峰值后的第一个显著波谷
    """
    distances = np.array(distances).flatten()

    # 1. 构建高分辨率的密度函数
    x_range = np.linspace(distances.min(), distances.max(), 1000)

    # 使用更小的带宽获得更精细的密度估计
    bandwidth = (distances.max() - distances.min()) / 100
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(distances.reshape(-1, 1))
    log_density = kde.score_samples(x_range.reshape(-1, 1))
    density = np.exp(log_density)

    # 2. 平滑密度函数
    density_smooth = gaussian_filter1d(density, sigma=2)

    # 3. 找到所有的峰值和谷值
    peaks, peak_properties = find_peaks(density_smooth, height=density_smooth.max() * 0.1, distance=20)
    valleys, valley_properties = find_peaks(-density_smooth, distance=10)

    print(f"🔍 检测到 {len(peaks)} 个峰值，{len(valleys)} 个谷值")

    if len(peaks) == 0:
        print("⚠️ 未检测到峰值，使用中位数")
        return np.median(distances)

    # 4. 找到第一个主要峰值（密度最高的前几个峰值之一）
    first_major_peak_idx = peaks[0]  # 通常第一个峰值就是主峰

    # 如果有多个峰值，选择密度最高的作为主峰
    if len(peaks) > 1:
        peak_heights = density_smooth[peaks]
        # 找到密度最高的峰值
        highest_peak_idx = peaks[np.argmax(peak_heights)]
        # 如果最高峰在前30%位置，则选择它；否则选择第一个峰
        if x_range[highest_peak_idx] < (distances.max() - distances.min()) * 0.3 + distances.min():
            first_major_peak_idx = highest_peak_idx

    first_major_peak_x = x_range[first_major_peak_idx]

    # 5. 在第一个主峰之后找到第一个显著的波谷
    # 筛选在主峰之后的谷值
    valleys_after_peak = valleys[valleys > first_major_peak_idx]

    if len(valleys_after_peak) == 0:
        print(f"⚠️ 主峰后无明显谷值，使用主峰位置 + 偏移")
        # 如果没有明显谷值，使用主峰后的最小密度点
        after_peak_region = density_smooth[first_major_peak_idx:]
        min_idx_relative = np.argmin(after_peak_region)
        threshold_idx = first_major_peak_idx + min_idx_relative
    else:
        # 6. 在主峰后的谷值中找到第一个"显著"的波谷
        # 显著性判断：谷值深度要足够（相对于周围峰值）
        best_valley_idx = None
        min_depth_ratio = 0.3  # 谷值深度至少是峰值的30%

        for valley_idx in valleys_after_peak:
            valley_density = density_smooth[valley_idx]
            peak_density = density_smooth[first_major_peak_idx]

            # 计算深度比例
            depth_ratio = 1 - (valley_density / peak_density)

            # 如果谷值足够深，且位置合理（不要太靠右）
            valley_position_ratio = (x_range[valley_idx] - distances.min()) / (distances.max() - distances.min())
            if depth_ratio >= min_depth_ratio and valley_position_ratio < 0.7:
                best_valley_idx = valley_idx
                break

        if best_valley_idx is not None:
            threshold_idx = best_valley_idx
        else:
            # 如果没找到符合条件的谷值，选择第一个谷值
            threshold_idx = valleys_after_peak[0]

    # 7. 转换回原始距离值
    threshold_value = x_range[threshold_idx]

    # 8. 合理性检查：阈值不应该太小或太大
    min_reasonable = distances.min() + (distances.max() - distances.min()) * 0.05
    max_reasonable = distances.min() + (distances.max() - distances.min()) * 0.5

    if threshold_value < min_reasonable:
        print(f"⚠️ 检测到的阈值过小，调整到最小合理值")
        threshold_value = min_reasonable
    elif threshold_value > max_reasonable:
        print(f"⚠️ 检测到的阈值过大，调整到最大合理值")
        threshold_value = max_reasonable

    print(f"🎯 梯度Gap检测阈值: {threshold_value:.4f}")
    print(f"📍 第一个主峰位置: {first_major_peak_x:.4f}")
    print(f"📉 选择的谷值位置: {threshold_value:.4f}")

    return threshold_value




def find_first_valley_threshold(distances, bins=100, smooth_window=5):
    """
    找到第一个波谷作为阈值
    """
    distances = np.array(distances).flatten()

    # 1. 构建直方图
    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. 平滑直方图（可选，减少噪声）
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        hist_smooth = uniform_filter1d(hist.astype(float), size=smooth_window)
    else:
        hist_smooth = hist.astype(float)

    # 3. 找所有峰值
    peaks, _ = find_peaks(hist_smooth, height=1)  # 至少高度为1

    # 4. 找所有谷值（负峰值）
    valleys, _ = find_peaks(-hist_smooth)

    print(f"🔍 检测到 {len(peaks)} 个峰值，{len(valleys)} 个谷值")

    if len(peaks) == 0:
        print("⚠️ 未检测到明显峰值，使用均值作为阈值")
        return np.mean(distances)

    if len(valleys) == 0:
        print("⚠️ 未检测到明显谷值，使用第一个峰值后的最小值")
        first_peak_idx = peaks[0]
        # 在第一个峰值后找最小值
        after_peak = hist_smooth[first_peak_idx:]
        if len(after_peak) > 1:
            min_idx_relative = np.argmin(after_peak)
            threshold_idx = first_peak_idx + min_idx_relative
        else:
            threshold_idx = first_peak_idx + 1
    else:
        # 5. 找第一个有意义的谷值
        # 要求：在第一个峰值之后，且不能太靠近边界
        first_peak_idx = peaks[0] if len(peaks) > 0 else 0

        valid_valleys = []
        for valley_idx in valleys:
            # 必须在第一个峰值之后
            if valley_idx > first_peak_idx:
                # 不能太靠近右边界
                if valley_idx < len(hist_smooth) * 0.8:
                    valid_valleys.append(valley_idx)

        if len(valid_valleys) > 0:
            threshold_idx = valid_valleys[0]  # 第一个有效谷值
        else:
            print("⚠️ 未找到有效谷值，使用第一个峰值后的最小值")
            first_peak_idx = peaks[0]
            after_peak = hist_smooth[first_peak_idx:]
            min_idx_relative = np.argmin(after_peak)
            threshold_idx = first_peak_idx + min_idx_relative

    # 6. 转换回原始距离值
    threshold_idx = min(threshold_idx, len(bin_centers) - 1)
    threshold_value = bin_centers[threshold_idx]

    print(f"🎯 第一个波谷阈值: {threshold_value:.4f}")
    print(f"📍 阈值位置: bin {threshold_idx}/{len(bin_centers)}")

    return threshold_value


def normalize_distances(distances):
    """
    将距离归一化到0-1之间
    """
    distances = np.array(distances)

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    if max_dist == min_dist:
        print("⚠️ 所有距离相同，归一化后全为0")
        return np.zeros_like(distances), min_dist, max_dist

    normalized = (distances - min_dist) / (max_dist - min_dist)

    print(f"📏 距离归一化: [{min_dist:.4f}, {max_dist:.4f}] -> [0.0, 1.0]")

    return normalized, min_dist, max_dist


def denormalize_threshold(normalized_threshold, min_dist, max_dist):
    """
    将归一化后的阈值转换回原始尺度
    """
    original_threshold = normalized_threshold * (max_dist - min_dist) + min_dist
    return original_threshold


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import warnings

warnings.filterwarnings("ignore")


def normalize_distances_minmax(distances):
    """Min-Max归一化距离"""
    distances = np.array(distances)
    min_val = np.min(distances)
    max_val = np.max(distances)

    if max_val - min_val == 0:
        return np.zeros_like(distances), min_val, max_val

    normalized = (distances - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def find_gradient_descent_threshold(distances, bins=100, smooth_sigma=1.0):
    """梯度下降检测：找到第一个显著负梯度点"""
    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 平滑直方图
    smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

    # 计算一阶导数（梯度）
    gradient = np.gradient(smoothed_hist)

    # 找到第一个显著负梯度点（下降幅度超过某个阈值）
    # 先找到峰值位置
    peak_idx = np.argmax(smoothed_hist)

    # 在峰值后寻找第一个显著下降点
    threshold_ratio = 0.1  # 下降幅度阈值（相对于最大梯度）
    max_neg_gradient = np.min(gradient)
    significant_drop = max_neg_gradient * threshold_ratio

    for i in range(peak_idx + 1, len(gradient)):
        if gradient[i] < significant_drop:
            return bin_centers[i]

    # 如果没找到，返回峰值后的第一个下降点
    for i in range(peak_idx + 1, len(gradient)):
        if gradient[i] < 0:
            return bin_centers[i]

    return bin_centers[peak_idx]

import numpy as np
from scipy.stats import gaussian_kde

import numpy as np
from scipy import stats


def find_gaussian_tail_threshold(distances, n_std=2.0, fit_percentile=25):
    """
    针对左侧高斯分布的尾部阈值检测

    Parameters:
    -----------
    distances : array-like
        距离数据
    n_std : float
        标准差倍数 (2.0对应约95%, 3.0对应约99.7%)
    fit_percentile : float
        用于拟合的数据百分位数，只用前X%的数据拟合以减少右侧异常值影响

    Returns:
    --------
    float : 阈值位置
    """
    distances = np.array(distances)

    # 只使用左侧数据进行高斯拟合，避免右侧异常值的影响
    fit_data = distances[distances <= np.percentile(distances, fit_percentile)]

    # 拟合高斯分布
    mu, sigma = stats.norm.fit(fit_data)

    # 计算n个标准差的阈值
    threshold = mu + n_std * sigma

    return threshold


import numpy as np
from scipy.stats import gaussian_kde


def find_simple_valley_threshold(distances, bins=150):
    """
    最简单的波谷检测：直接用直方图，不要KDE
    """
    # 直接用直方图，不要平滑
    counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 找最高峰
    peak_idx = np.argmax(counts)

    # 从峰值后找第一个局部最小值
    for i in range(peak_idx + 1, len(counts) - 1):
        if counts[i - 1] > counts[i] < counts[i + 1]:
            return bin_centers[i]

    return bin_centers[peak_idx + 10]  # fallback


def plot_distance_distribution_improved(distances, labels, num_known, threshold, dataset_name,
                                        use_normalized=True, save_path=None):
    """改进的距离分布图：纵坐标为数量，横坐标可选归一化"""

    # 数据预处理
    if use_normalized:
        distances_plot, min_val, max_val = normalize_distances_minmax(distances)
        threshold_plot = (threshold - min_val) / (max_val - min_val) if max_val != min_val else 0
        xlabel = f'Normalized Min Distance (Original: [{min_val:.4f}, {max_val:.4f}])'
    else:
        distances_plot = distances
        threshold_plot = threshold
        xlabel = 'Min Distance to Atoms'

    plt.figure(figsize=(15, 8))

    # 分离known和unknown样本的距离
    known_mask = labels != num_known
    unknown_mask = labels == num_known

    known_distances = distances_plot[known_mask]
    unknown_distances = distances_plot[unknown_mask]

    # 绘制直方图（使用count而不是density）
    plt.hist(known_distances, bins=50, alpha=0.7, label=f'Known Classes (n={len(known_distances)})',
             color='blue', density=False)
    plt.hist(unknown_distances, bins=50, alpha=0.7, label=f'Unknown Classes (n={len(unknown_distances)})',
             color='red', density=False)

    # 添加统计信息
    known_mean = np.mean(distances[known_mask]) if known_mask.sum() > 0 else 0
    known_std = np.std(distances[known_mask]) if known_mask.sum() > 0 else 0
    unknown_mean = np.mean(distances[unknown_mask]) if unknown_mask.sum() > 0 else 0
    unknown_std = np.std(distances[unknown_mask]) if unknown_mask.sum() > 0 else 0

    plt.axvline(np.mean(known_distances), color='blue', linestyle='--', alpha=0.8,
                label=f'Known Mean: {known_mean:.6f}±{known_std:.6f}')
    plt.axvline(np.mean(unknown_distances), color='red', linestyle='--', alpha=0.8,
                label=f'Unknown Mean: {unknown_mean:.6f}±{unknown_std:.6f}')

    # 添加阈值线
    plt.axvline(threshold_plot, color='green', linestyle='-', linewidth=3, alpha=0.9,
                label=f'Threshold: {threshold:.6f}')

    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(f'{dataset_name} - Distribution of Minimum Distance to Atoms (Count-based)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加统计文本
    stats_text = (f'Dataset: {dataset_name}\n'
                  f'Known: μ={known_mean:.6f}, σ={known_std:.6f}\n'
                  f'Unknown: μ={unknown_mean:.6f}, σ={unknown_std:.6f}\n'
                  f'Separation: {abs(unknown_mean - known_mean):.6f}\n'
                  f'Threshold: {threshold:.6f}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()