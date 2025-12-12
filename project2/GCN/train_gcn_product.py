import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm

# --- 1. 定义GCN模型 (适用于全图训练) ---
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

    # forward 方法适配全图 edge_index
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# --- 2. 本地数据加载函数 (新增采样功能) ---
def load_graph_data_from_csv(data_dir, device, subsample_ratio=0.01):
    """
    从本地CSV文件加载图数据, 并根据 subsample_ratio 进行采样。
    """
    print(f"--- 开始从 '{data_dir}' 加载图数据 (转导式) ---")
    
    try:
        full_x = torch.tensor(pd.read_csv(os.path.join(data_dir, 'node_feat.csv'), header=None).values, dtype=torch.float32)
        # product 是单标签分类，使用 long 类型
        full_y = torch.tensor(pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None).values, dtype=torch.long)
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
            
            subset = torch.randperm(num_nodes_full)[:num_nodes_sample]
            subset = torch.sort(subset).values

            edge_index, _ = subgraph(subset, full_edge_index, relabel_nodes=True, num_nodes=num_nodes_full)
            x = full_x[subset]
            y = full_y[subset]

            node_map = torch.full((num_nodes_full,), -1, dtype=torch.long)
            node_map[subset] = torch.arange(num_nodes_sample)
            
            split_idx = {}
            for split in ['train', 'valid', 'test']:
                mask = torch.isin(full_split_idx[split], subset)
                split_idx[split] = node_map[full_split_idx[split][mask]]
            
            print(f"采样完成。新图: {x.size(0):,} 个节点, {edge_index.size(1):,} 条边。")
            print(f"  - Train: {split_idx['train'].size(0):,}, Valid: {split_idx['valid'].size(0):,}, Test: {split_idx['test'].size(0):,}")
        else:
            print("未进行采样，使用完整图。")
            x, y, edge_index, split_idx = full_x, full_y, full_edge_index, full_split_idx

        # 将采样后的数据整个移动到GPU
        data = Data(x=x, edge_index=edge_index, y=y).to(device)
        num_classes = len(torch.unique(y.squeeze()))
        
        print("\n数据准备完毕。")
        print(f"  - 节点数: {data.num_nodes:,}, 边数: {data.num_edges:,}")
        print(f"  - 节点特征维度: {data.num_features}, 类别数量: {num_classes}")

        return data, split_idx, num_classes

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        return None, None, -1

# --- 3. 定义训练和评估函数 (全图版本) ---
def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1)
    y_true = data.y.squeeze(1)

    accs = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        if len(idx) == 0:
            accs[split] = 0.0
            continue
        correct = (y_pred[idx] == y_true[idx]).sum()
        accs[split] = int(correct) / int(idx.size(0))
    return accs['train'], accs['valid'], accs['test']

# --- 4. 结果保存与绘图函数 ---
def plot_curves(results_dir, epochs, train_losses, valid_accs):
    os.makedirs(results_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss Curve'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, valid_accs, label='Validation Accuracy', color='tab:blue')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.set_title('Validation Accuracy Curve'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves_gcn_product_sampled.png')
    plt.savefig(save_path)
    print(f"\n训练曲线图已保存到: {save_path}")

def save_summary(results_dir, best_valid_acc, test_acc, ratio):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'results_gcn_product_sampled.txt')
    with open(summary_path, 'w') as f:
        f.write(f"GCN Model for ogbn-products (Sampled at {ratio*100:.1f}%)\n")
        f.write("====================================================\n")
        f.write(f"Best Validation Accuracy: {best_valid_acc:.4f}\n")
        f.write(f"Test Accuracy on Best Model: {test_acc:.4f}\n")
    print(f"结果摘要已保存到: {summary_path}")

# --- 5. 主执行函数 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/product'
    results_dir = '/home/zhouruiqi/project/HW2/GCN/results'
    model_results_dir = os.path.join(results_dir, 'model_weights')
    best_model_path = os.path.join(model_results_dir, 'gcn_product_sampled.pth')
    os.makedirs(model_results_dir, exist_ok=True)

    # ogbn-products 图非常大，采样率需要设得更低
    subsample_ratio = 0.25 # 使用 30% 的数据

    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    lr = 1e-2
    epochs = 500
    eval_steps =10

    # 数据在加载时就进行采样并移动到 device
    data, split_idx, num_classes = load_graph_data_from_csv(data_dir, device, subsample_ratio=subsample_ratio)
    if data is None: return

    model = GCN(
        in_channels=data.num_features, hidden_channels=hidden_channels,
        out_channels=num_classes, num_layers=num_layers, dropout=dropout
    ).to(device) # 模型也移动到 device
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_valid_acc = 0
    best_test_acc = 0
    epoch_list, loss_list, acc_list = [], [], []

    print("\n--- 开始全图训练 (在采样图上) ---")
    pbar = tqdm(range(1, epochs + 1), desc="Training Epochs")
    for epoch in pbar:
        loss = train(model, data, split_idx['train'], optimizer, loss_fn)
        
        if epoch % eval_steps == 0:
            train_acc, valid_acc, test_acc = test(model, data, split_idx)
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                torch.save(model.state_dict(), best_model_path)
            
            epoch_list.append(epoch)
            loss_list.append(loss)
            acc_list.append(valid_acc)
            
            pbar.set_postfix(loss=f'{loss:.4f}', valid_acc=f'{valid_acc:.4f}')

    pbar.close()
    print("\n--- 训练完成 ---")
    print(f'最佳验证集准确率: {best_valid_acc:.4f}')
    print(f'对应的测试集准确率: {best_test_acc:.4f}')

    plot_curves(results_dir, epoch_list, loss_list, acc_list)
    save_summary(results_dir, best_valid_acc, best_test_acc, subsample_ratio)

if __name__ == "__main__":
    main()