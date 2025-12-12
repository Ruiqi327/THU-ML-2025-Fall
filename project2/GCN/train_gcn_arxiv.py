import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm

# --- 1. 定义GCN模型 ---
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

# --- 2. 本地数据加载函数 (适用于GCN的转导式设置) ---
def load_graph_data_from_csv(data_dir, device):
    """从本地CSV文件加载完整的图数据"""
    print(f"--- 开始从 '{data_dir}' 加载图数据 (转导式) ---")
    
    try:
        # 加载所有节点的特征
        node_feat = pd.read_csv(os.path.join(data_dir, 'node_feat.csv'), header=None).values
        x = torch.tensor(node_feat, dtype=torch.float32)

        # 加载所有节点的标签
        node_label = pd.read_csv(os.path.join(data_dir, 'node_label.csv'), header=None).values
        y = torch.tensor(node_label, dtype=torch.long)

        # 加载完整的边列表
        edges = pd.read_csv(os.path.join(data_dir, 'edge.csv'), header=None).values
        # PyG 需要 [2, num_edges] 的格式，所以需要转置
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 将有向图转为无向图 (GCN通常在无向图上表现更好)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # 加载划分索引
        split_idx = {}
        for split in ['train', 'valid', 'test']:
            idx = pd.read_csv(os.path.join(data_dir, f'{split}_idx.csv'), header=None).values.flatten()
            split_idx[split] = torch.tensor(idx, dtype=torch.long)

        # 创建 PyG 的 Data 对象
        data = Data(x=x, edge_index=edge_index, y=y).to(device)
        
        num_classes = len(torch.unique(y))
        print("数据加载完毕。")
        print(f"  - 节点数: {data.num_nodes:,}, 边数: {data.num_edges:,}")
        print(f"  - 节点特征维度: {data.num_features}, 类别数量: {num_classes}")

        return data, split_idx, num_classes

    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 -> {e.filename}")
        print("请确保以下文件都存在于指定目录中：node_feat.csv, node_label.csv, edge.csv, train_idx.csv, valid_idx.csv, test_idx.csv")
        return None, None, -1
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return None, None, -1

# --- 3. 定义训练和评估函数 ---
def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    # GCN在整个图上进行前向传播
    out = model(data.x, data.edge_index)
    # 只在训练节点上计算损失
    loss = loss_fn(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1)

    accs = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        correct = (y_pred[idx] == data.y.squeeze(1)[idx]).sum()
        accs[split] = int(correct) / int(idx.size(0))
    return accs['train'], accs['valid'], accs['test']

# --- 4. 结果保存与绘图函数 ---
def plot_curves(results_dir, epochs, train_losses, valid_accs):
    os.makedirs(results_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, valid_accs, label='Validation Accuracy', color='tab:blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Curve')
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves_gcn_arxiv.png')
    plt.savefig(save_path)
    print(f"\n训练曲线图已保存到: {save_path}")

def save_summary(results_dir, best_valid_acc, test_acc):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'results_gcn_arxiv.txt')
    
    with open(summary_path, 'w') as f:
        f.write("GCN Model for ogbn-arxiv\n")
        f.write("==========================\n")
        f.write(f"Best Validation Accuracy: {best_valid_acc:.4f}\n")
        f.write(f"Test Accuracy on Best Model: {test_acc:.4f}\n")
    print(f"结果摘要已保存到: {summary_path}")

# --- 5. 主执行函数 ---
def main():
    # --- 参数设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/arxiv'
    results_dir = '/home/zhouruiqi/project/HW2/GCN/results'
    model_results_dir = os.path.join(results_dir, 'model_weights')
    best_model_path = os.path.join(model_results_dir, 'gcn_arxiv.pth')

    os.makedirs(model_results_dir, exist_ok=True)

    # 模型参数
    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    
    # 训练参数
    lr = 0.01
    epochs = 300
    eval_steps = 5

    # --- 加载数据 ---
    data, split_idx, num_classes = load_graph_data_from_csv(data_dir, device)
    if data is None:
        return

    # --- 初始化模型、优化器和损失函数 ---
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- 开始训练和评估循环 ---
    best_valid_acc = 0
    best_test_acc = 0
    epoch_list, loss_list, acc_list = [], [], []

    print("\n--- 开始训练 ---")
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
    print(f"最佳验证集准确率: {best_valid_acc:.4f}")
    print(f"对应的测试集准确率: {best_test_acc:.4f}")

    # --- 在测试集上评估最佳模型 ---
    print("\n--- 开始在测试集上评估最佳模型 ---")
    best_model = GCN(
        in_channels=data.num_features, hidden_channels=hidden_channels,
        out_channels=num_classes, num_layers=num_layers, dropout=dropout
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    _, _, final_test_acc = test(best_model, data, split_idx)
    print(f"加载的最佳模型在测试集上的准确率: {final_test_acc:.4f}")

    # --- 保存结果和绘图 ---
    plot_curves(results_dir, epoch_list, loss_list, acc_list)
    save_summary(results_dir, best_valid_acc, final_test_acc)

if __name__ == "__main__":
    main()