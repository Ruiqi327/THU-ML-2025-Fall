import os
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据加载和图构建 ---
def load_data_and_build_adj(data_dir):
    """
    为 ogbn-arxiv 加载数据并构建邻接表。
    """
    print(f"--- 开始从 '{data_dir}' 加载数据 ---")
    try:
        # 加载边，用于构建图
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), header=None)
        
        # 加载标签和划分索引
        y_true_df = pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None)
        train_idx = pd.read_csv(os.path.join(data_dir, 'train_idx.csv'), header=None)[0].values
        valid_idx = pd.read_csv(os.path.join(data_dir, 'valid_idx.csv'), header=None)[0].values
        test_idx = pd.read_csv(os.path.join(data_dir, 'test_idx.csv'), header=None)[0].values

        num_nodes = len(y_true_df)
        y_true = torch.from_numpy(y_true_df.values).squeeze(1)

        print(f"数据加载完毕。节点数: {num_nodes:,}, 边数: {len(edge_df):,}")

        # 构建邻接表
        print("正在构建邻接表...")
        adj = defaultdict(list)
        # ogbn-arxiv 是有向图，LPA 通常在无向图上效果更好，所以我们构建无向邻接表
        for src, dst in tqdm(edge_df.values, desc="Building Adjacency List"):
            adj[src].append(dst)
            adj[dst].append(src)
        
        # 将邻居列表转换为Tensor，以便后续在GPU上操作
        print("正在将邻接表转换为Tensor...")
        adj_tensor = {node: torch.tensor(neighbors, dtype=torch.long) for node, neighbors in adj.items()}

        return adj_tensor, y_true, train_idx, valid_idx, test_idx, num_nodes

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        return None, None, None, None, None, -1

# --- 2. 标签传播算法 (LPA) 核心实现 (增加验证过程) ---
@torch.no_grad()
def run_lpa(adj, y_true, train_idx, valid_idx, num_nodes, max_iter, device):
    print("\n--- 开始执行标签传播算法 (LPA) ---")
    # 初始化：未知节点的标签等于其节点ID，训练节点的标签等于真实标签
    y_pred = torch.arange(num_nodes, dtype=torch.long)
    y_pred[train_idx] = y_true[train_idx]
    y_pred = y_pred.to(device)
    
    # 将真实标签也移动到设备上，以便高效计算验证准确率
    y_true_device = y_true.to(device)

    validation_history = []

    # --- 在迭代前计算初始准确率 (第0次迭代) ---
    correct = (y_pred[valid_idx] == y_true_device[valid_idx]).sum().item()
    initial_acc = correct / len(valid_idx)
    validation_history.append(initial_acc)
    print(f"Initial State (Iteration 0) Validation Accuracy: {initial_acc:.4f}")

    for node, neighbors in adj.items():
        adj[node] = neighbors.to(device)

    for i in range(max_iter):
        changed = False
        nodes_to_update = np.random.permutation(num_nodes)
        
        pbar = tqdm(nodes_to_update, desc=f"LPA Iteration {i+1}/{max_iter}", leave=False)
        for node in pbar:
            # 训练集节点的标签是固定的，不应被更新
            if node in train_idx:
                continue

            neighbors = adj.get(node)
            if neighbors is None or len(neighbors) == 0:
                continue

            neighbor_labels = y_pred[neighbors]
            if len(neighbor_labels) == 0: continue
            
            # 找出出现次数最多的标签
            counts = torch.bincount(neighbor_labels)
            new_label = torch.argmax(counts)

            if y_pred[node] != new_label:
                y_pred[node] = new_label
                changed = True
        
        # --- 在每次迭代后计算验证集准确率 ---
        correct = (y_pred[valid_idx] == y_true_device[valid_idx]).sum().item()
        acc = correct / len(valid_idx)
        validation_history.append(acc)
        print(f"Iteration {i+1}/{max_iter} 完成。Validation Accuracy: {acc:.4f}")
        
        if not changed:
            print("标签已收敛，提前停止。")
            # 如果提前停止，用最后一次的准确率填充剩余的迭代历史
            remaining_iters = max_iter - (i + 1)
            if remaining_iters > 0:
                validation_history.extend([acc] * remaining_iters)
            break
            
    return y_pred.cpu(), validation_history

# --- 3. 评估、绘图和保存结果 ---
def evaluate(y_pred, y_true, split_idx, split_name):
    idx = split_idx[split_name]
    correct = (y_pred[idx] == y_true[idx]).sum()
    acc = int(correct) / len(idx)
    print(f"{split_name.capitalize()} Accuracy: {acc:.4f}")
    return acc

def plot_and_save_history(history, results_dir):
    """绘制并保存验证准确率随迭代次数变化的曲线图"""
    os.makedirs(results_dir, exist_ok=True)
    
    # x轴从0开始，代表初始状态
    iterations = range(len(history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history, marker='o', linestyle='-')
    plt.title('LPA Validation Accuracy per Iteration on ogbn-arxiv')
    plt.xlabel('Iteration (0 = Initial State)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    
    # 确保x轴刻度是整数
    if len(history) <= 21: # 如果总迭代次数（包括0）不多，显示所有刻度
        plt.xticks(iterations)
    
    # 找到最佳准确率并标注
    best_acc = max(history)
    best_iter = history.index(best_acc)
    plt.axvline(x=best_iter, color='r', linestyle='--', label=f'Best Acc: {best_acc:.4f} at Iter {best_iter}')
    plt.legend()
    
    save_path = os.path.join(results_dir, 'lpa_arxiv_validation_accuracy.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\n验证准确率曲线图已保存到: {save_path}")

def save_summary(results_dir, valid_acc, test_acc):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'results_lpa_arxiv.txt')
    with open(summary_path, 'w') as f:
        f.write("Label Propagation (LPA) for ogbn-arxiv\n")
        f.write("======================================\n")
        f.write(f"Validation Accuracy: {valid_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
    print(f"结果摘要已保存到: {summary_path}")

# --- 4. 主执行函数 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    # 修改数据路径
    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/arxiv'
    results_dir = '/home/zhouruiqi/project/HW2/LPA/results'
    
    # LPA 参数
    max_iterations = 5 # arxiv 图结构可能更复杂，可以适当增加迭代次数

    # 加载数据
    adj, y_true, train_idx, valid_idx, test_idx, num_nodes = load_data_and_build_adj(data_dir)
    if adj is None: return

    # 运行 LPA
    y_pred, validation_history = run_lpa(adj, y_true, train_idx, valid_idx, num_nodes, max_iterations, device)

    # 评估最终结果
    print("\n--- 评估最终结果 ---")
    split_idx_map = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    evaluate(y_pred, y_true, split_idx_map, 'train')
    valid_acc = evaluate(y_pred, y_true, split_idx_map, 'valid')
    test_acc = evaluate(y_pred, y_true, split_idx_map, 'test')

    # 绘图和保存摘要
    plot_and_save_history(validation_history, results_dir)
    save_summary(results_dir, valid_acc, test_acc)

if __name__ == "__main__":
    main()