import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- 1. 定义GCN模型 (保持不变) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# --- 2. 本地数据加载函数 (新增采样功能) ---
def load_graph_data_from_csv(data_dir, device, subsample_ratio=0.1):
    """
    从本地CSV文件加载完整的图数据, 并根据 subsample_ratio 进行采样。
    """
    print(f"--- 开始从 '{data_dir}' 加载图数据 (转导式) ---")
    
    try:
        full_x = torch.tensor(pd.read_csv(os.path.join(data_dir, 'node_feat.csv'), header=None).values, dtype=torch.float32)
        full_y = torch.tensor(pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None).values, dtype=torch.float32)
        full_edge_index = torch.tensor(pd.read_csv(os.path.join(data_dir, 'edge.csv'), header=None).values, dtype=torch.long).t().contiguous()
        full_edge_index = torch.cat([full_edge_index, full_edge_index.flip(0)], dim=1)

        full_split_idx = {}
        for split in ['train', 'valid', 'test']:
            idx = pd.read_csv(os.path.join(data_dir, f'{split}_idx.csv'), header=None).values.flatten()
            full_split_idx[split] = torch.tensor(idx, dtype=torch.long)

        num_nodes_full = full_x.size(0)
        print(f"完整图加载完毕: {num_nodes_full:,} 个节点, {full_edge_index.size(1):,} 条边。")

        if subsample_ratio < 1.0:
            print(f"\n--- 开始进行图采样 (Ratio: {subsample_ratio}) ---")
            num_nodes_sample = int(num_nodes_full * subsample_ratio)
            
            # 1. 随机选择子图节点
            subset = torch.randperm(num_nodes_full)[:num_nodes_sample]
            subset = torch.sort(subset).values # 排序以保持相对顺序

            # 2. 创建诱导子图
            edge_index, _ = subgraph(subset, full_edge_index, relabel_nodes=True, num_nodes=num_nodes_full)
            x = full_x[subset]
            y = y = full_y[subset]

            # 3. 更新 split_idx
            # 创建一个映射，将旧索引映射到新索引
            node_map = torch.full((num_nodes_full,), -1, dtype=torch.long)
            node_map[subset] = torch.arange(num_nodes_sample)
            
            split_idx = {}
            for split in ['train', 'valid', 'test']:
                # 筛选出在子图中的节点，并重新映射索引
                mask = torch.isin(full_split_idx[split], subset)
                split_idx[split] = node_map[full_split_idx[split][mask]]
            
            print(f"采样完成。新图: {x.size(0):,} 个节点, {edge_index.size(1):,} 条边。")
            print(f"  - Train: {split_idx['train'].size(0):,}, Valid: {split_idx['valid'].size(0):,}, Test: {split_idx['test'].size(0):,}")

        else:
            print("未进行采样，使用完整图。")
            x, y, edge_index, split_idx = full_x, full_y, full_edge_index, full_split_idx

        data = Data(x=x, edge_index=edge_index, y=y).to(device)
        num_tasks = y.size(1)
        
        print("\n数据准备完毕。")
        print(f"  - 节点数: {data.num_nodes:,}, 边数: {data.num_edges:,}")
        print(f"  - 节点特征维度: {data.num_features}, 任务(标签)数量: {num_tasks}")

        return data, split_idx, num_tasks

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        return None, None, -1
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return None, None, -1

# --- 3. 定义训练和评估函数 (适配 proteins) ---
def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx):
    model.eval()
    out = model(data.x, data.edge_index)
    
    y_true = data.y.cpu().numpy()
    y_pred = out.cpu().numpy()

    aucs = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        if len(idx) == 0: # 如果某个划分集在采样后为空
            aucs[split] = 0.0
            continue
        aucs[split] = roc_auc_score(y_true[idx], y_pred[idx])
        
    return aucs['train'], aucs['valid'], aucs['test']

# --- 4. 结果保存与绘图函数 (适配 proteins) ---
def plot_curves(results_dir, epochs, train_losses, valid_aucs):
    os.makedirs(results_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss Curve'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, valid_aucs, label='Validation ROC-AUC', color='tab:blue')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('ROC-AUC'); ax2.set_title('Validation ROC-AUC Curve'); ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves_gcn_proteins_sampled.png')
    plt.savefig(save_path)
    print(f"\n训练曲线图已保存到: {save_path}")

def save_summary(results_dir, best_valid_auc, test_auc, ratio):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'results_gcn_proteins_sampled.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"GCN Model for ogbn-proteins (Sampled at {ratio*100:.0f}%)\n")
        f.write("==================================================\n")
        f.write(f"Best Validation ROC-AUC: {best_valid_auc:.4f}\n")
        f.write(f"Test ROC-AUC on Best Model: {test_auc:.4f}\n")
    print(f"结果摘要已保存到: {summary_path}")

# --- 5. 主执行函数 ---
def main():
    # --- 参数设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/proteins'
    results_dir = '/home/zhouruiqi/project/HW2/GCN/results'
    model_results_dir = os.path.join(results_dir, 'model_weights')
    best_model_path = os.path.join(model_results_dir, 'gcn_proteins_sampled.pth')

    os.makedirs(model_results_dir, exist_ok=True)

    # 新增：采样率
    subsample_ratio = 0.2 # 使用20%的数据

    # 模型参数
    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    
    # 训练参数
    lr = 0.01
    epochs = 800
    eval_steps = 10

    # --- 加载数据 ---
    data, split_idx, num_tasks = load_graph_data_from_csv(data_dir, device, subsample_ratio=subsample_ratio)
    if data is None:
        return

    # --- 初始化模型、优化器和损失函数 ---
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=num_tasks,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # --- 开始训练和评估循环 ---
    best_valid_auc = 0
    best_test_auc = 0
    epoch_list, loss_list, auc_list = [], [], []

    print("\n--- 开始训练 (在采样图上) ---")
    pbar = tqdm(range(1, epochs + 1), desc="Training Epochs")
    for epoch in pbar:
        loss = train(model, data, split_idx['train'], optimizer, loss_fn)
        
        if epoch % eval_steps == 0:
            train_auc, valid_auc, test_auc = test(model, data, split_idx)
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_test_auc = test_auc
                torch.save(model.state_dict(), best_model_path)
            
            epoch_list.append(epoch)
            loss_list.append(loss)
            auc_list.append(valid_auc)
            
            pbar.set_postfix(loss=f'{loss:.4f}', valid_auc=f'{valid_auc:.4f}')

    pbar.close()
    print("\n--- 训练完成 ---")
    print(f"最佳验证集 ROC-AUC: {best_valid_auc:.4f}")
    print(f"对应的测试集 ROC-AUC: {best_test_auc:.4f}")

    # --- 保存结果和绘图 ---
    plot_curves(results_dir, epoch_list, loss_list, auc_list)
    save_summary(results_dir, best_valid_auc, best_test_auc, subsample_ratio)

if __name__ == "__main__":
    main()