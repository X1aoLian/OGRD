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
    import random
    print('---------------------------------------------------------------')
    tmp = TUDataset(root, dataset_name, use_node_attr=True)

    if tmp.num_node_features == 0:
        max_deg = 0
        for data in tmp:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_deg = max(max_deg, int(deg.max()))
        dataset = TUDataset(root, dataset_name, transform=OneHotDegree(max_deg))
    else:
        dataset = tmp

    labels = torch.tensor([int(data.y) for data in dataset])

    all_classes = sorted(labels.unique().tolist())
    known_classes = [c for c in all_classes if c not in unknown_classes]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")
    print(f"Data scale factor: {scale_factor}")

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling TU dataset (scale factor: {scale_factor}) ===")

        all_class_indices = {}
        for idx in range(len(dataset)):
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

                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        dataset = scaled_dataset
        labels = scaled_labels

        print(f"Scaled dataset size: {len(dataset)}")

    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== TU dataset stratified sampling details ===")
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

            print(f"Original class {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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
            if self.is_unknown:
                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
            else:
                try:
                    if data.y.numel() == 1:
                        original_label = data.y.item()
                    else:
                        original_label = data.y.flatten()[0].item()
                except:
                    original_label = int(data.y.cpu().numpy().flatten()[0])

                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    num_known = len(known_classes)

    train_known = RelabeledSubset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledSubset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledSubset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    print(f"\n=== Verifying TU dataset training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_dataset_with_class_merging(
        dataset_name: str,
        class_mapping: dict,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        **kwargs
):
    import random
    import torch

    print(f"=== Class merging dataset split: {dataset_name} ===")
    print(f"Class mapping scheme: {class_mapping}")

    if 'unknown' not in class_mapping:
        raise ValueError("class_mapping must contain 'unknown' key to specify unknown classes")

    unknown_classes = class_mapping['unknown']

    known_mapping = {}
    for new_label, old_labels in class_mapping.items():
        if new_label != 'unknown':
            for old_label in old_labels:
                known_mapping[old_label] = new_label

    all_involved_classes = []
    for old_labels in class_mapping.values():
        all_involved_classes.extend(old_labels)

    print(f"Original unknown classes: {unknown_classes}")
    print(f"Known class mapping: {known_mapping}")
    print(f"All involved original classes: {sorted(all_involved_classes)}")

    temp_train, temp_test, temp_unknown = load_dataset_unified(
        dataset_name, [], ratio, root, seed, scale_factor=scale_factor, **kwargs
    )

    all_data_indices = list(range(len(temp_train) + len(temp_test) + len(temp_unknown)))
    all_dataset = []
    all_labels = []

    for dataset_part in [temp_train, temp_test, temp_unknown]:
        for data in dataset_part:
            all_dataset.append(data)
            all_labels.append(data.y.item())

    print(f"Collected total data: {len(all_dataset)}")
    print(f"Original label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    filtered_dataset = []
    filtered_labels = []

    for i, (data, label) in enumerate(zip(all_dataset, all_labels)):
        if label in all_involved_classes:
            filtered_dataset.append(data)
            filtered_labels.append(label)

    print(f"Filtered data count: {len(filtered_dataset)}")
    print(f"Filtered label distribution: {dict(zip(*np.unique(filtered_labels, return_counts=True)))}")

    merged_dataset = []
    merged_labels = []

    for data, old_label in zip(filtered_dataset, filtered_labels):
        new_data = data.clone()

        if old_label in unknown_classes:
            new_label = -1
        else:
            if old_label in known_mapping:
                new_label = known_mapping[old_label]
            else:
                print(f"Warning: label {old_label} not in known mapping, skipping")
                continue

        new_data.y = torch.tensor(new_label, dtype=data.y.dtype, device=data.y.device)
        merged_dataset.append(new_data)
        merged_labels.append(new_label)

    known_labels = [label for label in merged_labels if label != -1]
    unknown_count = sum(1 for label in merged_labels if label == -1)

    print(f"\n=== Class merging results ===")
    if known_labels:
        known_distribution = dict(zip(*np.unique(known_labels, return_counts=True)))
        print(f"Known class distribution: {known_distribution}")
    print(f"Unknown class count: {unknown_count}")

    num_known = len(set(known_labels)) if known_labels else 0

    for data, label in zip(merged_dataset, merged_labels):
        if label == -1:
            data.y = torch.tensor(num_known, dtype=data.y.dtype, device=data.y.device)

    merged_labels = [num_known if label == -1 else label for label in merged_labels]

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling dataset (scale factor: {scale_factor}) ===")

        class_indices = {}
        for idx, label in enumerate(merged_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        scaled_indices = []
        for class_label, indices in class_indices.items():
            original_count = len(indices)
            scaled_count = max(1, int(original_count * scale_factor))

            random.shuffle(indices)
            sampled_indices = indices[:scaled_count]
            scaled_indices.extend(sampled_indices)

            class_name = f"Class {class_label}" if class_label != num_known else "Unknown"
            print(f"{class_name}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [merged_dataset[i] for i in scaled_indices]
        scaled_labels = [merged_labels[i] for i in scaled_indices]

        merged_dataset = scaled_dataset
        merged_labels = scaled_labels

        print(f"Scaled dataset size: {len(merged_dataset)}")

    unknown_indices = [i for i, label in enumerate(merged_labels) if label == num_known]
    known_indices = [i for i, label in enumerate(merged_labels) if label != num_known]

    class_indices = {}
    for idx in known_indices:
        label = merged_labels[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== Stratified sampling details ===")
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

        print(f"Merged class {class_label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)
    random.shuffle(unknown_indices)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_indices)}")

    if len(unknown_indices) > test_known_size:
        unknown_indices = unknown_indices[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_indices) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_indices)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_indices)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_indices)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_indices)}")

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

    print(f"\n=== Verifying merged training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        label = merged_labels[idx]
        train_label_count[label] = train_label_count.get(label, 0) + 1

    print(f"Training set label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def example_usage():
    class_mapping_mnist = {
        0: [0, 1],
        1: [2, 3, 4],
        2: [5, 6, 7],
        'unknown': [8, 9]
    }

    class_mapping_5to2 = {
        0: [0, 1],
        1: [2, 3],
        'unknown': [4]
    }

    class_mapping_10to3 = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7],
        'unknown': [8, 9]
    }

    print("=== Usage Examples ===")
    print("Example 1 - MNIST 10 -> 3 classes:")
    print(f"  class_mapping = {class_mapping_mnist}")
    print("\nExample 2 - 5 classes -> 2 classes:")
    print(f"  class_mapping = {class_mapping_5to2}")
    print("\nExample 3 - 10 classes -> 3 classes:")
    print(f"  class_mapping = {class_mapping_10to3}")


def split_mnist_dataset(unknown_classes, ratio, root='../data', seed=42, scale_factor=0.01):
    import random
    import torch

    train_dataset = MNISTSuperpixels(root=root + '/MNISTSuperpixels', train=True)
    test_dataset = MNISTSuperpixels(root=root + '/MNISTSuperpixels', train=False)

    all_dataset = list(train_dataset) + list(test_dataset)

    print(f"MNIST dataset original total size: {len(all_dataset)}")
    print(f"Node feature dimension: {all_dataset[0].x.shape[1]}")
    print(f"Total classes: 10 (digits 0-9)")
    print(f"Data scale factor: {scale_factor}")

    labels = torch.tensor([int(data.y) for data in all_dataset])

    all_classes = list(range(10))
    known_classes = [c for c in all_classes if c not in unknown_classes]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")

    random.seed(seed)

    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling dataset (scale factor: {scale_factor}) ===")

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

                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        all_dataset = scaled_dataset
        labels = scaled_labels

        print(f"Scaled dataset size: {len(all_dataset)}")

    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== MNIST stratified sampling details ===")
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

            print(f"Digit {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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
            if self.is_unknown:
                data.y = torch.tensor(self.num_known, dtype=data.y.dtype, device=data.y.device)
            else:
                original_label = data.y.item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=data.y.dtype, device=data.y.device)
            return data

    num_known = len(known_classes)

    train_known = RelabeledMNISTSubset(all_dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledMNISTSubset(all_dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledMNISTSubset(all_dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    print(f"\n=== Verifying MNIST training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_colors3_dataset(
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0
):
    import random
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import OneHotDegree
    from torch_geometric.utils import degree

    dataset_name = 'COLORS-3'

    tmp = TUDataset(root + '/TUDataset', dataset_name, use_node_attr=True)

    if tmp.num_node_features == 0:
        max_deg = 0
        for data in tmp:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_deg = max(max_deg, int(deg.max()))
        dataset = TUDataset(root + '/TUDataset', dataset_name, transform=OneHotDegree(max_deg))
    else:
        dataset = tmp

    print(f"COLORS-3 dataset original total size: {len(dataset)}")
    print(
        f"Node feature dimension: {dataset[0].x.shape[1] if hasattr(dataset[0], 'x') and dataset[0].x is not None else 'No node features'}")
    print(f"Data scale factor: {scale_factor}")

    labels = []
    for data in dataset:
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        labels.append(int(label))
    labels = torch.tensor(labels, dtype=torch.long)

    all_classes = sorted(labels.unique().tolist())
    known_classes = [c for c in all_classes if c not in unknown_classes]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling COLORS-3 dataset (scale factor: {scale_factor}) ===")

        all_class_indices = {}
        for idx in range(len(dataset)):
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

                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices], dtype=torch.long)

        dataset = scaled_dataset
        labels = scaled_labels

        print(f"Scaled dataset size: {len(dataset)}")

    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== COLORS-3 stratified sampling details ===")
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

            print(f"Class {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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

            if self.is_unknown:
                data.y = torch.tensor(self.num_known, dtype=torch.long, device=data.y.device)
            else:
                original_label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
                original_label = int(original_label)

                if original_label in self.label_mapping:
                    new_label = self.label_mapping[original_label]
                    data.y = torch.tensor(new_label, dtype=torch.long, device=data.y.device)

            return data

    num_known = len(known_classes)

    train_known = RelabeledCOLORS3Subset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = RelabeledCOLORS3Subset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = RelabeledCOLORS3Subset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    print(f"\n=== Verifying COLORS-3 training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_reddit_multi_5k_dataset(
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        max_degree: int = 20
):
    import random
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import degree, one_hot

    dataset_name = 'REDDIT-MULTI-5K'

    print(f"=== Loading REDDIT-MULTI-5K dataset ===")
    print(f"Max degree limit: {max_degree} (feature dimension: {max_degree + 1})")
    print(f"Data scale factor: {scale_factor}")

    class SimpleClampedDegree:
        def __init__(self, max_deg):
            self.max_deg = max_deg

        def __call__(self, data):
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            deg = torch.clamp(deg, 0, self.max_deg)
            data.x = one_hot(deg, num_classes=self.max_deg + 1).float()
            return data

    dataset = TUDataset(
        root + '/TUDataset',
        dataset_name,
        transform=SimpleClampedDegree(max_degree)
    )

    print(f"Original dataset size: {len(dataset)}")
    print(f"Node feature dimension: {max_degree + 1}")

    labels = []
    for data in dataset:
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        labels.append(int(label))
    labels = torch.tensor(labels, dtype=torch.long)

    all_classes = [0, 1, 2, 3, 4]
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling dataset (scale factor: {scale_factor}) ===")

        class_indices = {}
        for idx, label in enumerate(labels):
            label = label.item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        scaled_indices = []
        for class_label in all_classes:
            if class_label in class_indices:
                indices = class_indices[class_label]
                original_count = len(indices)
                scaled_count = max(1, int(original_count * scale_factor))

                random.shuffle(indices)
                sampled_indices = indices[:scaled_count]
                scaled_indices.extend(sampled_indices)

                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])

        dataset = scaled_dataset
        labels = scaled_labels
        print(f"Scaled dataset size: {len(dataset)}")

    unknown_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== REDDIT-MULTI-5K stratified sampling details ===")
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

            print(f"Class {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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
                data.y = torch.tensor(self.num_known, dtype=torch.long, device=data.y.device)
            else:
                original_label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
                if original_label in self.label_mapping:
                    data.y = torch.tensor(self.label_mapping[original_label], dtype=torch.long, device=data.y.device)

            return data

    num_known = len(known_classes)

    train_known = SimpleRedditSubset(dataset, train_idx, label_mapping, is_unknown=False)
    test_known = SimpleRedditSubset(dataset, test_idx, label_mapping, is_unknown=False)
    all_unknown = SimpleRedditSubset(dataset, unknown_idx, label_mapping, is_unknown=True, num_known=num_known)

    print(f"\n=== Verifying REDDIT-MULTI-5K training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        if original_label in label_mapping:
            remapped_label = label_mapping[original_label]
            train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1

    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def load_dataset_unified(dataset_name, unknown_classes, ratio, root, seed, **kwargs):
    scale_factor = kwargs.get('scale_factor', 1.0)

    if dataset_name == 'REDDIT-MULTI-5K':
        max_degree = kwargs.get('max_degree', 50)
        return split_reddit_multi_5k_dataset(unknown_classes, ratio, root, seed, scale_factor, max_degree)

    if dataset_name == 'MNIST':
        return split_mnist_dataset(unknown_classes, ratio, root, seed, scale_factor)
    elif dataset_name == 'COLORS-3':
        return split_colors3_dataset(unknown_classes, ratio, root, seed, scale_factor)
    elif dataset_name in ['CLUSTER', 'CSL', 'CIFAR10']:
        return split_gnn_benchmark_dataset(dataset_name, unknown_classes, ratio, root, seed, scale_factor)

    elif dataset_name in ['10', '40']:
        num_points = kwargs.get('num_points', 1024)
        k_neighbors = kwargs.get('k_neighbors', 6)
        return split_modelnet_dataset(dataset_name, unknown_classes, ratio, root, seed,
                                      scale_factor, num_points, k_neighbors)

    else:
        return split_dataset(dataset_name, unknown_classes, ratio, root, seed, scale_factor)


def split_gnn_benchmark_dataset(
        dataset_name: str,
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0
):
    from torch_geometric.datasets import GNNBenchmarkDataset
    import torch
    import random

    non_binary_datasets = {
        'CLUSTER': 6,
        'CSL': 10,
        'MNIST': 10,
        'CIFAR10': 10
    }

    binary_datasets = ['PATTERN', 'TSP']

    if dataset_name in binary_datasets:
        raise ValueError(f"Dataset {dataset_name} is a binary classification dataset, not supported. "
                         f"Please use non-binary datasets: {list(non_binary_datasets.keys())}")

    if dataset_name not in non_binary_datasets:
        raise ValueError(f"Unsupported dataset {dataset_name}. "
                         f"Supported datasets: {list(non_binary_datasets.keys())}")

    print(f"=== Loading GNNBenchmarkDataset: {dataset_name} ===")

    train_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='train')
    val_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='val')
    test_dataset = GNNBenchmarkDataset(root=root + '/GNNBenchmark', name=dataset_name, split='test')

    all_dataset = list(train_dataset) + list(val_dataset) + list(test_dataset)
    num_classes = non_binary_datasets[dataset_name]

    print(f"Original dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Merged total size: {len(all_dataset)}")
    print(
        f"Node feature dimension: {all_dataset[0].x.shape[1] if hasattr(all_dataset[0], 'x') and all_dataset[0].x is not None else 'No node features'}")
    print(f"Total classes: {num_classes}")
    print(f"Data scale factor: {scale_factor}")

    labels = torch.tensor([data.y.item() if data.y.numel() == 1 else data.y[0].item() for data in all_dataset])

    all_classes = list(range(num_classes))
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling dataset (scale factor: {scale_factor}) ===")
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
                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])
        all_dataset = scaled_dataset
        labels = scaled_labels
        print(f"Scaled dataset size: {len(all_dataset)}")

    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== {dataset_name} stratified sampling details ===")
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
            print(f"Class {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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

    print(f"\n=== Verifying {dataset_name} training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1
    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

    return train_known, test_known, all_unknown


def split_modelnet_dataset(
        dataset_name: str,
        unknown_classes: list,
        ratio: float,
        root: str = '../data',
        seed: int = 42,
        scale_factor: float = 1.0,
        num_points: int = 1024,
        k_neighbors: int = 6
):
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, KNNGraph, Compose
    import torch
    import random

    if dataset_name not in ["10", "40"]:
        raise ValueError(f"ModelNet dataset name must be '10' or '40', got: {dataset_name}")

    print(f"=== Loading ModelNet{dataset_name} dataset ===")

    class AddPositionAsFeature:
        def __call__(self, data):
            data.x = data.pos
            return data

    transform = Compose([
        SamplePoints(num_points, remove_faces=False, include_normals=False),
        KNNGraph(k=k_neighbors),
        AddPositionAsFeature()
    ])

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

    print(f"Original dataset sizes: train={len(train_dataset)}, test={len(test_dataset)}")
    print(f"Merged total size: {len(all_dataset)}")
    print(f"Point cloud points: {num_points}")
    print(f"k-NN neighbors: {k_neighbors}")
    print(f"Node feature dimension: 3 (x, y, z coordinates)")
    print(f"Total classes: {num_classes}")
    print(f"Data scale factor: {scale_factor}")

    labels = torch.tensor([int(data.y) for data in all_dataset])

    all_classes = list(range(num_classes))
    known_classes = [c for c in all_classes if c not in unknown_classes]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}

    print(f"Original classes: {all_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {known_classes}")
    print(f"Label mapping: {label_mapping}")

    random.seed(seed)
    if scale_factor < 1.0:
        print(f"\n=== Proportionally scaling dataset (scale factor: {scale_factor}) ===")
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
                print(f"Class {class_label}: {original_count} -> {scaled_count} samples")

        scaled_dataset = [all_dataset[i] for i in scaled_indices]
        scaled_labels = torch.tensor([labels[i] for i in scaled_indices])
        all_dataset = scaled_dataset
        labels = scaled_labels
        print(f"Scaled dataset size: {len(all_dataset)}")

    unknown_mask = torch.zeros(len(all_dataset), dtype=torch.bool)
    for c in unknown_classes:
        unknown_mask |= (labels == c)
    unknown_idx = unknown_mask.nonzero(as_tuple=False).view(-1).tolist()

    class_indices = {}
    for idx in range(len(all_dataset)):
        if not unknown_mask[idx]:
            label = labels[idx].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    print(f"\n=== ModelNet{dataset_name} stratified sampling details ===")
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
            print(f"Class {label}: total={n_samples}, train={n_train}, test={n_test}")

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    test_known_size = len(test_idx)
    print(f"Test set (known classes) sample count: {test_known_size}")
    print(f"Original unknown sample count: {len(unknown_idx)}")

    if len(unknown_idx) > test_known_size:
        random.shuffle(unknown_idx)
        unknown_idx = unknown_idx[:test_known_size]
        print(f"Randomly sampled {test_known_size} from unknown samples")
    elif len(unknown_idx) < test_known_size:
        print(f"Warning: unknown sample count ({len(unknown_idx)}) less than test set size ({test_known_size})")
        print(f"Keeping all {len(unknown_idx)} unknown samples")
    else:
        print(f"Unknown sample count matches test set: {len(unknown_idx)}")

    print(f"Total training samples: {len(train_idx)}")
    print(f"Total test samples: {len(test_idx)}")
    print(f"Final unknown samples: {len(unknown_idx)}")

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

    print(f"\n=== Verifying ModelNet{dataset_name} training set class distribution ===")
    train_label_count = {}
    for idx in train_idx:
        original_label = labels[idx].item()
        remapped_label = label_mapping[original_label]
        train_label_count[remapped_label] = train_label_count.get(remapped_label, 0) + 1
    print(f"Training set remapped label distribution: {dict(sorted(train_label_count.items()))}")

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
            C, K, D = initial_atoms.shape
            if gamma is None:
                self.gamma = 1.0 / D
            else:
                self.gamma = gamma
        else:
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
            mmd_distances = torch.zeros(bs, C, device=device)
            for i in range(bs):
                Xi = x_mat[batch == i]
                for j in range(C):
                    Aj = self.atoms[j]
                    mmd_distances[i, j] = self._mmd(Xi, Aj)
            return mmd_distances

        else:
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
        floor_idx = indices.floor().long().clamp(0, current_size - 1)
        ceil_idx = indices.ceil().long().clamp(0, current_size - 1)
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


import torch


def sample_noise_features(model, data_batch, ratio=0.3, distance_factor=5.0, noise_sigma=3.0):
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    all_node_features = model.extract_node_features(data_batch)

    neg_features_list = []
    neg_batch_list = []

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        graph_features = all_node_features[mask]

        neg_features = graph_features.clone()

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
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    all_node_features = model.extract_node_features(data_batch)
    atoms = model.atoms
    num_classes, num_atom_supp, hidden_dim = atoms.shape

    gamma = model.gamma

    positive_distances = []

    for i in range(batch_size):
        mask = data_batch.batch == i
        graph_features = all_node_features[mask]
        graph_label = data_batch.y[i].item()

        target_atom = atoms[graph_label]
        mmd_dist = compute_mmd_your_way(graph_features, target_atom, gamma)
        positive_distances.append(mmd_dist)

    baseline_distance = np.mean(positive_distances)
    threshold = baseline_distance * multiplier

    atoms_center = atoms.mean(dim=(0, 1))

    all_features_norm = torch.norm(all_node_features, dim=1).mean().item()
    feature_std = all_node_features.std().item()

    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    neg_features_list = []
    neg_batch_list = []

    success_count = 0

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        original_features = all_node_features[mask]
        num_nodes = original_features.size(0)

        graph_center = original_features.mean(dim=0)

        direction_to_graph = graph_center - atoms_center
        direction_norm = direction_to_graph.norm()

        if direction_norm > 1e-8:
            push_direction = direction_to_graph / direction_norm
        else:
            push_direction = torch.randn_like(graph_center)
            push_direction = push_direction / push_direction.norm()

        found_valid = False

        for attempt in range(max_attempts):
            push_distance = distance_step * (attempt + 1)

            directed_push = push_direction.unsqueeze(0).repeat(num_nodes, 1) * push_distance

            random_noise = torch.randn_like(original_features) * feature_std * 0.1

            perturbed_features = original_features + directed_push + random_noise

            min_mmd_to_atoms = float('inf')

            for class_idx in range(num_classes):
                current_atom = atoms[class_idx]
                mmd_dist = compute_mmd_your_way(perturbed_features, current_atom, gamma)
                min_mmd_to_atoms = min(min_mmd_to_atoms, mmd_dist)

            if min_mmd_to_atoms >= threshold:
                neg_features_list.append(perturbed_features)

                graph_batch = torch.full((num_nodes,), new_graph_id,
                                         dtype=torch.long, device=device)
                neg_batch_list.append(graph_batch)

                found_valid = True
                success_count += 1

                break

        if not found_valid:
            max_push_distance = distance_step * max_attempts
            directed_push = push_direction.unsqueeze(0).repeat(num_nodes, 1) * max_push_distance
            random_noise = torch.randn_like(original_features) * feature_std * 0.1
            perturbed_features = original_features + directed_push + random_noise

            min_mmd_to_atoms = float('inf')
            for class_idx in range(num_classes):
                current_atom = atoms[class_idx]
                mmd_dist = compute_mmd_your_way(perturbed_features, current_atom, gamma)
                min_mmd_to_atoms = min(min_mmd_to_atoms, mmd_dist)

            neg_features_list.append(perturbed_features)

            graph_batch = torch.full((num_nodes,), new_graph_id,
                                     dtype=torch.long, device=device)
            neg_batch_list.append(graph_batch)

    if len(neg_features_list) > 0:
        x_neg = torch.cat(neg_features_list, dim=0)
        neg_batch = torch.cat(neg_batch_list, dim=0)
    else:
        x_neg = torch.empty(0, hidden_dim, device=device)
        neg_batch = torch.empty(0, dtype=torch.long, device=device)

    success_rate = success_count / len(selected_indices) if selected_indices else 0

    return x_neg, neg_batch


def compute_mmd_your_way(x: torch.Tensor, y: torch.Tensor, gamma: float) -> float:
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(2)
    k_xx = torch.exp(-gamma * xx).mean()

    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    k_yy = torch.exp(-gamma * yy).mean()

    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    k_xy = torch.exp(-gamma * xy).mean()

    mmd_result = k_xx + k_yy - 2 * k_xy

    return mmd_result.item()


def sample_neg_features_improved(model, data_batch, selected_indices, sigma=2.0, p_drop=0.3):
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

        noise = torch.randn_like(graph_features) * sigma

        flip_mask = torch.rand_like(graph_features) < 0.3
        noise[flip_mask] *= -1

        atoms_center = model.atoms.mean(dim=(0, 1))
        push_direction = torch.randn_like(atoms_center)
        push_direction = push_direction / push_direction.norm()

        neg_features = graph_features + noise + push_direction.unsqueeze(0) * 3.0

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


def sample_neg_features_atoms(model, data_batch, ratio=0.3, distance_factor=5.0, noise_sigma=3.0):
    device = data_batch.x.device
    batch_size = int(data_batch.batch.max()) + 1

    num_selected = max(1, int(batch_size * ratio))
    selected_indices = torch.randperm(batch_size)[:num_selected].tolist()

    all_node_features = model.extract_node_features(data_batch)

    atoms_center = model.atoms.mean(dim=(0, 1))

    neg_features_list = []
    neg_batch_list = []

    for new_graph_id, orig_graph_idx in enumerate(selected_indices):
        mask = data_batch.batch == orig_graph_idx
        graph_features = all_node_features[mask]

        graph_center = graph_features.mean(dim=0)
        direction = graph_center - atoms_center
        direction = direction / (direction.norm() + 1e-8)

        neg_center = atoms_center - direction * distance_factor

        neg_features = neg_center.unsqueeze(0).repeat(graph_features.size(0), 1)
        neg_features += torch.randn_like(neg_features) * noise_sigma

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


import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter, find_peaks

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter, find_peaks

import numpy as np


def find_gradient_gap_threshold(distances):
    distances = np.array(distances).flatten()

    x_range = np.linspace(distances.min(), distances.max(), 1000)

    bandwidth = (distances.max() - distances.min()) / 100
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(distances.reshape(-1, 1))
    log_density = kde.score_samples(x_range.reshape(-1, 1))
    density = np.exp(log_density)

    density_smooth = gaussian_filter1d(density, sigma=2)

    peaks, peak_properties = find_peaks(density_smooth, height=density_smooth.max() * 0.1, distance=20)
    valleys, valley_properties = find_peaks(-density_smooth, distance=10)

    print(f"Detected {len(peaks)} peaks, {len(valleys)} valleys")

    if len(peaks) == 0:
        print("Warning: no peaks detected, using median")
        return np.median(distances)

    first_major_peak_idx = peaks[0]

    if len(peaks) > 1:
        peak_heights = density_smooth[peaks]
        highest_peak_idx = peaks[np.argmax(peak_heights)]
        if x_range[highest_peak_idx] < (distances.max() - distances.min()) * 0.3 + distances.min():
            first_major_peak_idx = highest_peak_idx

    first_major_peak_x = x_range[first_major_peak_idx]

    valleys_after_peak = valleys[valleys > first_major_peak_idx]

    if len(valleys_after_peak) == 0:
        print(f"Warning: no obvious valleys after main peak, using main peak position + offset")
        after_peak_region = density_smooth[first_major_peak_idx:]
        min_idx_relative = np.argmin(after_peak_region)
        threshold_idx = first_major_peak_idx + min_idx_relative
    else:
        best_valley_idx = None
        min_depth_ratio = 0.3

        for valley_idx in valleys_after_peak:
            valley_density = density_smooth[valley_idx]
            peak_density = density_smooth[first_major_peak_idx]

            depth_ratio = 1 - (valley_density / peak_density)

            valley_position_ratio = (x_range[valley_idx] - distances.min()) / (distances.max() - distances.min())
            if depth_ratio >= min_depth_ratio and valley_position_ratio < 0.7:
                best_valley_idx = valley_idx
                break

        if best_valley_idx is not None:
            threshold_idx = best_valley_idx
        else:
            threshold_idx = valleys_after_peak[0]

    threshold_value = x_range[threshold_idx]

    min_reasonable = distances.min() + (distances.max() - distances.min()) * 0.05
    max_reasonable = distances.min() + (distances.max() - distances.min()) * 0.5

    if threshold_value < min_reasonable:
        print(f"Warning: detected threshold too small, adjusting to minimum reasonable value")
        threshold_value = min_reasonable
    elif threshold_value > max_reasonable:
        print(f"Warning: detected threshold too large, adjusting to maximum reasonable value")
        threshold_value = max_reasonable

    print(f"Gradient gap detection threshold: {threshold_value:.4f}")
    print(f"First main peak position: {first_major_peak_x:.4f}")
    print(f"Selected valley position: {threshold_value:.4f}")

    return threshold_value


def find_first_valley_threshold(distances, bins=100, smooth_window=5):
    distances = np.array(distances).flatten()

    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        hist_smooth = uniform_filter1d(hist.astype(float), size=smooth_window)
    else:
        hist_smooth = hist.astype(float)

    peaks, _ = find_peaks(hist_smooth, height=1)

    valleys, _ = find_peaks(-hist_smooth)

    print(f"Detected {len(peaks)} peaks, {len(valleys)} valleys")

    if len(peaks) == 0:
        print("Warning: no obvious peaks detected, using mean as threshold")
        return np.mean(distances)

    if len(valleys) == 0:
        print("Warning: no obvious valleys detected, using minimum after first peak")
        first_peak_idx = peaks[0]
        after_peak = hist_smooth[first_peak_idx:]
        if len(after_peak) > 1:
            min_idx_relative = np.argmin(after_peak)
            threshold_idx = first_peak_idx + min_idx_relative
        else:
            threshold_idx = first_peak_idx + 1
    else:
        first_peak_idx = peaks[0] if len(peaks) > 0 else 0

        valid_valleys = []
        for valley_idx in valleys:
            if valley_idx > first_peak_idx:
                if valley_idx < len(hist_smooth) * 0.8:
                    valid_valleys.append(valley_idx)

        if len(valid_valleys) > 0:
            threshold_idx = valid_valleys[0]
        else:
            print("Warning: no valid valley found, using minimum after first peak")
            first_peak_idx = peaks[0]
            after_peak = hist_smooth[first_peak_idx:]
            min_idx_relative = np.argmin(after_peak)
            threshold_idx = first_peak_idx + min_idx_relative

    threshold_idx = min(threshold_idx, len(bin_centers) - 1)
    threshold_value = bin_centers[threshold_idx]

    print(f"First valley threshold: {threshold_value:.4f}")
    print(f"Threshold position: bin {threshold_idx}/{len(bin_centers)}")

    return threshold_value


def normalize_distances(distances):
    distances = np.array(distances)

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    if max_dist == min_dist:
        print("Warning: all distances equal, normalization results in zeros")
        return np.zeros_like(distances), min_dist, max_dist

    normalized = (distances - min_dist) / (max_dist - min_dist)

    print(f"Distance normalization: [{min_dist:.4f}, {max_dist:.4f}] -> [0.0, 1.0]")

    return normalized, min_dist, max_dist


def denormalize_threshold(normalized_threshold, min_dist, max_dist):
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
    distances = np.array(distances)
    min_val = np.min(distances)
    max_val = np.max(distances)

    if max_val - min_val == 0:
        return np.zeros_like(distances), min_val, max_val

    normalized = (distances - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def find_gradient_descent_threshold(distances, bins=100, smooth_sigma=1.0):
    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

    gradient = np.gradient(smoothed_hist)

    peak_idx = np.argmax(smoothed_hist)

    threshold_ratio = 0.1
    max_neg_gradient = np.min(gradient)
    significant_drop = max_neg_gradient * threshold_ratio

    for i in range(peak_idx + 1, len(gradient)):
        if gradient[i] < significant_drop:
            return bin_centers[i]

    for i in range(peak_idx + 1, len(gradient)):
        if gradient[i] < 0:
            return bin_centers[i]

    return bin_centers[peak_idx]


import numpy as np
from scipy.stats import gaussian_kde

import numpy as np
from scipy import stats


def find_gaussian_tail_threshold(distances, n_std=2.0, fit_percentile=25):
    distances = np.array(distances)

    fit_data = distances[distances <= np.percentile(distances, fit_percentile)]

    mu, sigma = stats.norm.fit(fit_data)

    threshold = mu + n_std * sigma

    return threshold


import numpy as np
from scipy.stats import gaussian_kde


def find_simple_valley_threshold(distances, bins=150):
    counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_idx = np.argmax(counts)

    for i in range(peak_idx + 1, len(counts) - 1):
        if counts[i - 1] > counts[i] < counts[i + 1]:
            return bin_centers[i]

    return bin_centers[peak_idx + 10]


def plot_distance_distribution_improved(distances, labels, num_known, threshold, dataset_name,
                                        use_normalized=True, save_path=None):
    if use_normalized:
        distances_plot, min_val, max_val = normalize_distances_minmax(distances)
        threshold_plot = (threshold - min_val) / (max_val - min_val) if max_val != min_val else 0
        xlabel = f'Normalized Min Distance (Original: [{min_val:.4f}, {max_val:.4f}])'
    else:
        distances_plot = distances
        threshold_plot = threshold
        xlabel = 'Min Distance to Atoms'

    plt.figure(figsize=(15, 8))

    known_mask = labels != num_known
    unknown_mask = labels == num_known

    known_distances = distances_plot[known_mask]
    unknown_distances = distances_plot[unknown_mask]

    plt.hist(known_distances, bins=50, alpha=0.7, label=f'Known Classes (n={len(known_distances)})',
             color='blue', density=False)
    plt.hist(unknown_distances, bins=50, alpha=0.7, label=f'Unknown Classes (n={len(unknown_distances)})',
             color='red', density=False)

    known_mean = np.mean(distances[known_mask]) if known_mask.sum() > 0 else 0
    known_std = np.std(distances[known_mask]) if known_mask.sum() > 0 else 0
    unknown_mean = np.mean(distances[unknown_mask]) if unknown_mask.sum() > 0 else 0
    unknown_std = np.std(distances[unknown_mask]) if unknown_mask.sum() > 0 else 0

    plt.axvline(np.mean(known_distances), color='blue', linestyle='--', alpha=0.8,
                label=f'Known Mean: {known_mean:.6f}+/-{known_std:.6f}')
    plt.axvline(np.mean(unknown_distances), color='red', linestyle='--', alpha=0.8,
                label=f'Unknown Mean: {unknown_mean:.6f}+/-{unknown_std:.6f}')

    plt.axvline(threshold_plot, color='green', linestyle='-', linewidth=3, alpha=0.9,
                label=f'Threshold: {threshold:.6f}')

    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(f'{dataset_name} - Distribution of Minimum Distance to Atoms (Count-based)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    stats_text = (f'Dataset: {dataset_name}\n'
                  f'Known: mu={known_mean:.6f}, sigma={known_std:.6f}\n'
                  f'Unknown: mu={unknown_mean:.6f}, sigma={unknown_std:.6f}\n'
                  f'Separation: {abs(unknown_mean - known_mean):.6f}\n'
                  f'Threshold: {threshold:.6f}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
