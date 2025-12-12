import torch
import pandas as pd
import numpy as np
from ogb.nodeproppred import Evaluator
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_subgraph_data(data_dir, sample_frac, device):
    """
    高效加载数据并直接构建子图，避免为全图创建邻接表。
    """
    print(f"--- 开始高效加载数据并构建子图 (采样率: {sample_frac*100:.1f}%) ---")
    try:
        # 1. 加载原始数据
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), header=None)
        labels_df = pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None)
        train_idx = pd.read_csv(os.path.join(data_dir, 'train_idx.csv'), header=None).values.flatten()
        valid_idx = pd.read_csv(os.path.join(data_dir, 'valid_idx.csv'), header=None).values.flatten()
        test_idx = pd.read_csv(os.path.join(data_dir, 'test_idx.csv'), header=None).values.flatten()

        num_nodes_full = len(labels_df)
        print(f"完整图数据: {num_nodes_full:,} 个节点, {len(edge_df):,} 条边。")

        # 2. 采样节点
        num_sampled_nodes = int(num_nodes_full * sample_frac)
        sampled_nodes_idx = np.random.permutation(num_nodes_full)[:num_sampled_nodes]
        
        # 3. 高效筛选子图的边 (使用Pandas向量化操作)
        print("正在使用Pandas高效筛选子图的边...")
        src_nodes, dst_nodes = edge_df[0].values, edge_df[1].values
        is_in_subset = np.isin(src_nodes, sampled_nodes_idx) & np.isin(dst_nodes, sampled_nodes_idx)
        subgraph_edge_df = edge_df[is_in_subset]

        # 4. 重映射节点ID
        print("正在重映射节点ID...")
        sampled_nodes_idx = np.sort(sampled_nodes_idx)
        old_to_new_map = {old_id: new_id for new_id, old_id in enumerate(sampled_nodes_idx)}
        
        new_src = subgraph_edge_df[0].map(old_to_new_map).values
        new_dst = subgraph_edge_df[1].map(old_to_new_map).values

        # 5. 直接构建子图的邻接表
        print("正在直接构建子图的邻接表...")
        subgraph_adj = [[] for _ in range(num_sampled_nodes)]
        for u, v in zip(new_src, new_dst):
            subgraph_adj[u].append(v)
            subgraph_adj[v].append(u) # 构建无向图
        
        subgraph_adj_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in subgraph_adj]

        # 6. 准备子图的标签和数据集划分
        subgraph_labels = torch.tensor(labels_df.values[sampled_nodes_idx], dtype=torch.float).to(device)
        
        subgraph_train_idx = torch.tensor([old_to_new_map[idx] for idx in train_idx if idx in old_to_new_map], dtype=torch.long, device=device)
        subgraph_valid_idx = torch.tensor([old_to_new_map[idx] for idx in valid_idx if idx in old_to_new_map], dtype=torch.long, device=device)
        subgraph_test_idx = torch.tensor([old_to_new_map[idx] for idx in test_idx if idx in old_to_new_map], dtype=torch.long, device=device)

        print("子图数据准备完成。")
        print(f"  - 子图节点数: {num_sampled_nodes:,}, 子图边数: {len(subgraph_edge_df):,}")
        
        return subgraph_adj_tensor, subgraph_labels, subgraph_train_idx, subgraph_valid_idx, subgraph_test_idx

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        return None, None, None, None, None

def run_lpa(adj, labels, train_idx, valid_idx, num_iterations=10):
    """
    在给定的子图上运行多标签LPA，并在每次迭代后评估验证集。
    【已改回非锚定传播模式】
    """
    num_nodes, num_labels = labels.shape
    evaluator = Evaluator(name='ogbn-proteins')
    validation_history = []

    # 1. 初始化：使用 -1 作为未标记节点的哨兵值
    propagated_labels = torch.full_like(labels, -1.0)
    propagated_labels[train_idx] = labels[train_idx]

    predict_nodes_mask = torch.ones(num_nodes, dtype=torch.bool, device=labels.device)
    predict_nodes_mask[train_idx] = False
    predict_nodes_indices = torch.where(predict_nodes_mask)[0]

    print(f"\n开始在子图上运行LPA（非锚定模式，其中 {len(predict_nodes_indices)} 个节点需要预测标签）...")

    # --- 评估初始状态 (Iteration 0) ---
    # 对于评估，临时将-1替换为0
    y_true_valid = labels[valid_idx].cpu().numpy()
    y_pred_valid_eval = propagated_labels[valid_idx].clone().cpu().numpy()
    y_pred_valid_eval[y_pred_valid_eval == -1] = 0
    initial_rocauc = evaluator.eval({'y_true': y_true_valid, 'y_pred': y_pred_valid_eval})['rocauc']
    validation_history.append(initial_rocauc)
    print(f"Initial State (Iteration 0) Validation ROC-AUC: {initial_rocauc:.4f}")

    for iteration in range(num_iterations):
        start_time = time.time()
        total_changes = 0
        
        # 非锚定模式使用异步更新更符合经典LPA的逻辑
        permuted_indices = predict_nodes_indices[torch.randperm(len(predict_nodes_indices))]

        for node_idx_tensor in tqdm(permuted_indices, desc=f"迭代 {iteration + 1}/{num_iterations}", leave=False):
            node_idx = node_idx_tensor.item()
            neighbors = adj[node_idx]
            
            if len(neighbors) == 0: continue

            # --- 【修正核心】---
            # 1. 找出所有已被标记的邻居（标签不是-1），包括训练集和之前迭代中被预测的节点
            known_labels_mask = propagated_labels[neighbors, 0] != -1.0
            valid_neighbors = neighbors[known_labels_mask]

            if len(valid_neighbors) == 0: continue

            # 2. 从这些邻居的当前传播标签中收集信息
            neighbor_labels = propagated_labels[valid_neighbors]
            
            # 3. 逐元素进行多数投票
            label_counts = torch.sum(neighbor_labels, dim=0)
            threshold = len(valid_neighbors) / 2.0
            new_label_vector = (label_counts > threshold).float()
            
            # 检查标签是否改变，并直接更新（异步更新）
            if torch.any(propagated_labels[node_idx] != new_label_vector):
                total_changes += 1
                propagated_labels[node_idx] = new_label_vector
        
        iter_time = time.time() - start_time
        
        # --- 迭代后评估 ---
        y_true_valid = labels[valid_idx].cpu().numpy()
        y_pred_valid_eval = propagated_labels[valid_idx].clone().cpu().numpy()
        y_pred_valid_eval[y_pred_valid_eval == -1] = 0 # 替换-1用于评估
        valid_rocauc = evaluator.eval({'y_true': y_true_valid, 'y_pred': y_pred_valid_eval})['rocauc']
        validation_history.append(valid_rocauc)
        
        print(f"迭代 {iteration + 1} 完成，耗时: {iter_time:.2f}s, 标签改变节点数: {total_changes}, 验证集 ROC-AUC: {valid_rocauc:.4f}")

        if total_changes == 0:
            print("标签不再改变，提前停止。")
            remaining_iters = num_iterations - (iteration + 1)
            if remaining_iters > 0:
                validation_history.extend([valid_rocauc] * remaining_iters)
            break

    # 最终将所有未标记的节点（如果还有的话）设置为0向量
    propagated_labels[propagated_labels == -1] = 0
    return propagated_labels, validation_history

def plot_and_save_history(history, save_path):
    """绘制验证集准确率随迭代次数变化的曲线并保存。"""
    if not history:
        print("没有可供绘制的历史记录。")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    iterations = range(len(history))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history, 'b-o', label='Validation ROC-AUC')
    plt.xlabel('Iteration (0 = Initial State)', fontsize=14)
    plt.ylabel('ROC-AUC Score', fontsize=14)
    plt.title('Validation ROC-AUC per LPA Iteration (Non-Anchored)', fontsize=16)
    
    if len(history) <= 21:
        plt.xticks(iterations)

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"验证集性能曲线图已保存至: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/proteins'
    results_dir = '/home/zhouruiqi/project/HW2/LPA/results'
    sample_frac = 0.1
    num_lpa_iterations = 10

    subgraph_data = prepare_subgraph_data(data_dir, sample_frac, device)
    if subgraph_data[0] is None: return
    subgraph_adj, subgraph_labels, subgraph_train_idx, subgraph_valid_idx, subgraph_test_idx = subgraph_data

    final_labels, validation_history = run_lpa(subgraph_adj, subgraph_labels, subgraph_train_idx, subgraph_valid_idx, num_lpa_iterations)

    plot_filename = f"lpa_proteins_validation_rocauc_frac{int(sample_frac*100)}_non_anchored.png"
    plot_save_path = os.path.join(results_dir, plot_filename)
    plot_and_save_history(validation_history, plot_save_path)

    y_true_test = subgraph_labels[subgraph_test_idx].cpu().numpy()
    y_pred_test = final_labels[subgraph_test_idx].cpu().numpy()

    evaluator = Evaluator(name='ogbn-proteins')
    test_rocauc = evaluator.eval({'y_true': y_true_test, 'y_pred': y_pred_test})['rocauc']

    print(f"\n--- 在 {sample_frac*100:.1f}% 的子图上最终评估 ---")
    print(f"测试集 ROC-AUC: {test_rocauc:.4f}")

if __name__ == '__main__':
    main()