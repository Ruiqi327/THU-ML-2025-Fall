import os
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据加载和图构建 (新增采样功能) ---
def load_data_and_build_adj(data_dir, subsample_ratio=0.1):
    """
    加载数据，根据 subsample_ratio 进行采样，并构建邻接表。
    """
    print(f"--- 开始从 '{data_dir}' 加载数据 ---")
    try:
        # 加载完整图数据
        full_edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), header=None)
        full_y_df = pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None)
        full_train_idx = pd.read_csv(os.path.join(data_dir, 'train_idx.csv'), header=None)[0].values
        full_valid_idx = pd.read_csv(os.path.join(data_dir, 'valid_idx.csv'), header=None)[0].values
        full_test_idx = pd.read_csv(os.path.join(data_dir, 'test_idx.csv'), header=None)[0].values

        num_nodes_full = len(full_y_df)
        print(f"完整图加载完毕: {num_nodes_full:,} 个节点, {len(full_edge_df):,} 条边。")

        if subsample_ratio < 1.0:
            print(f"\n--- 开始进行图采样 (Ratio: {subsample_ratio}) ---")
            num_nodes_sample = int(num_nodes_full * subsample_ratio)
            
            # 1. 随机选择子图节点
            subset = np.random.permutation(num_nodes_full)[:num_nodes_sample]
            subset = np.sort(subset) # 排序以保持相对顺序

            # 2. 创建旧索引到新索引的映射
            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subset)}

            # 3. 筛选边并构建新邻接表
            print("正在构建采样后的小图邻接表...")
            adj = defaultdict(list)
            for src, dst in tqdm(full_edge_df.values, desc="Filtering Edges"):
                if src in node_map and dst in node_map:
                    new_src, new_dst = node_map[src], node_map[dst]
                    adj[new_src].append(new_dst)
                    adj[new_dst].append(new_src)
            
            # 4. 筛选标签和划分索引
            y_true = torch.from_numpy(full_y_df.values[subset]).squeeze(1)
            
            train_idx = [node_map[i] for i in full_train_idx if i in node_map]
            valid_idx = [node_map[i] for i in full_valid_idx if i in node_map]
            test_idx = [node_map[i] for i in full_test_idx if i in node_map]
            
            num_nodes = len(subset)
            print(f"采样完成。新图: {num_nodes:,} 个节点。")
            print(f"  - Train: {len(train_idx):,}, Valid: {len(valid_idx):,}, Test: {len(test_idx):,}")

        else: # 如果不采样
            raise NotImplementedError("当前脚本强制要求进行采样。")

        # 将邻居列表转换为Tensor
        print("正在将邻接表转换为Tensor...")
        adj_tensor = {node: torch.tensor(neighbors, dtype=torch.long) for node, neighbors in adj.items()}

        return adj_tensor, y_true, np.array(train_idx), np.array(valid_idx), np.array(test_idx), num_nodes

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        return None, None, None, None, None, -1

# --- 2. 标签传播算法 (LPA) 核心实现 (增加验证过程) ---
@torch.no_grad()
def run_lpa(adj, y_true, train_idx, valid_idx, num_nodes, max_iter, device):
    print("\n--- 开始执行标签传播算法 (LPA) ---")
    y_pred = torch.arange(num_nodes, dtype=torch.long)
    y_pred[train_idx] = y_true[train_idx]
    y_pred = y_pred.to(device)
    
    y_true_device = y_true.to(device)
    validation_history = []

    # --- 在迭代前计算初始准确率 (第0次迭代) ---
    if len(valid_idx) > 0:
        correct = (y_pred[valid_idx] == y_true_device[valid_idx]).sum().item()
        initial_acc = correct / len(valid_idx)
        validation_history.append(initial_acc)
        print(f"Initial State (Iteration 0) Validation Accuracy: {initial_acc:.4f}")
    else:
        validation_history.append(0.0) # 如果没有验证样本，则准确率为0
        print("Initial State (Iteration 0) Validation Accuracy: N/A (0 samples)")


    for node, neighbors in adj.items():
        adj[node] = neighbors.to(device)

    for i in range(max_iter):
        changed = False
        nodes_to_update = np.random.permutation(num_nodes)
        
        pbar = tqdm(nodes_to_update, desc=f"LPA Iteration {i+1}/{max_iter}", leave=False)
        for node in pbar:
            if node in train_idx:
                continue

            neighbors = adj.get(node)
            if neighbors is None or len(neighbors) == 0:
                continue

            neighbor_labels = y_pred[neighbors]
            if len(neighbor_labels) == 0: continue
            
            counts = torch.bincount(neighbor_labels)
            new_label = torch.argmax(counts)

            if y_pred[node] != new_label:
                y_pred[node] = new_label
                changed = True
        
        # --- 在每次迭代后计算验证集准确率 ---
        acc = 0.0
        if len(valid_idx) > 0:
            correct = (y_pred[valid_idx] == y_true_device[valid_idx]).sum().item()
            acc = correct / len(valid_idx)
        validation_history.append(acc)
        print(f"Iteration {i+1}/{max_iter} 完成。Validation Accuracy: {acc:.4f}")
        
        if not changed:
            print("标签已收敛，提前停止。")
            remaining_iters = max_iter - (i + 1)
            if remaining_iters > 0:
                validation_history.extend([acc] * remaining_iters)
            break
            
    return y_pred.cpu(), validation_history

# --- 3. 评估、绘图和保存结果 ---
def evaluate(y_pred, y_true, split_idx, split_name):
    idx = split_idx[split_name]
    if len(idx) == 0:
        print(f"{split_name.capitalize()} Accuracy: N/A (0 samples)")
        return 0.0
    correct = (y_pred[idx] == y_true[idx]).sum()
    acc = int(correct) / len(idx)
    print(f"{split_name.capitalize()} Accuracy: {acc:.4f}")
    return acc

def plot_and_save_history(history, results_dir, ratio):
    """绘制并保存验证准确率随迭代次数变化的曲线图"""
    os.makedirs(results_dir, exist_ok=True)
    
    iterations = range(len(history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history, marker='o', linestyle='-')
    plt.title(f'LPA Validation Accuracy on ogbn-products ({ratio*100:.0f}% Sample)')
    plt.xlabel('Iteration (0 = Initial State)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    
    if len(history) <= 21:
        plt.xticks(iterations)
    
    if any(history): # 仅当history不全为0时才标注
        best_acc = max(history)
        best_iter = history.index(best_acc)
        plt.axvline(x=best_iter, color='r', linestyle='--', label=f'Best Acc: {best_acc:.4f} at Iter {best_iter}')
        plt.legend()
    
    save_path = os.path.join(results_dir, f'lpa_product_validation_accuracy_{ratio*100:.0f}p.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\n验证准确率曲线图已保存到: {save_path}")

def save_summary(results_dir, valid_acc, test_acc, ratio):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f'results_lpa_product_sampled_{ratio*100:.0f}p.txt')
    with open(summary_path, 'w') as f:
        f.write(f"LPA for ogbn-products (Sampled at {ratio*100:.0f}%)\n")
        f.write("============================================\n")
        f.write(f"Validation Accuracy: {valid_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
    print(f"结果摘要已保存到: {summary_path}")

# --- 4. 主执行函数 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/product'
    results_dir = '/home/zhouruiqi/project/HW2/LPA/results'
    
    # --- 参数 ---
    subsample_ratio = 0.1 # 采样 10% 的数据
    max_iterations = 10

    # 加载并采样数据
    adj, y_true, train_idx, valid_idx, test_idx, num_nodes = load_data_and_build_adj(data_dir, subsample_ratio)
    if adj is None: return

    # 运行 LPA
    y_pred, validation_history = run_lpa(adj, y_true, train_idx, valid_idx, num_nodes, max_iterations, device)

    # 评估结果
    print("\n--- 评估最终结果 ---")
    split_idx_map = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    evaluate(y_pred, y_true, split_idx_map, 'train')
    valid_acc = evaluate(y_pred, y_true, split_idx_map, 'valid')
    test_acc = evaluate(y_pred, y_true, split_idx_map, 'test')

    # 绘图和保存摘要
    plot_and_save_history(validation_history, results_dir, subsample_ratio)
    save_summary(results_dir, valid_acc, test_acc, subsample_ratio)

if __name__ == "__main__":
    main()