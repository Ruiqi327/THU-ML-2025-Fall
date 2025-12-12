import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- 1. 定义MLP模型 ---
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

# --- 2. 数据加载函数 ---
def load_data_from_csv(data_dir, device):
    """从CSV文件加载特征和标签"""
    print(f"--- 开始从 '{data_dir}' 加载数据 ---")
    data = {}
    for split in ['train', 'valid', 'test']:
        feat_path = os.path.join(data_dir, f'{split}.csv')
        label_path = os.path.join(data_dir, f'{split}_label.csv')
        
        if not (os.path.exists(feat_path) and os.path.exists(label_path)):
            print(f"错误: {split} 的特征或标签文件不存在。")
            return None, -1

        features = pd.read_csv(feat_path, header=None).values
        labels = pd.read_csv(label_path, header=None).values
        
        data[f'x_{split}'] = torch.tensor(features, dtype=torch.float32).to(device)
        # 修改: 标签类型改为 float 以适应BCEWithLogitsLoss
        data[f'y_{split}'] = torch.tensor(labels, dtype=torch.float32).to(device)
        
        print(f"成功加载 {split} 数据: {data[f'x_{split}'].shape[0]} 个样本")

    # 修改: 从标签的形状获取任务数量 (112)
    num_tasks = data['y_train'].shape[1]
    print(f"数据加载完毕。任务数量: {num_tasks}")
    return data, num_tasks

# --- 3. 定义训练和评估函数 ---
def train(model, x_train, y_train, optimizer, loss_fn):
    """训练一个 epoch"""
    model.train()
    optimizer.zero_grad()
    out = model(x_train)
    # 修改: 直接传递多标签张量，不使用 .squeeze()
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, x, y):
    """在给定的数据集上评估模型，计算 ROC-AUC 分数"""
    model.eval()
    out = model(x)
    
    # 将真实标签和模型输出（logits）移动到CPU
    y_true = y.cpu().numpy()
    y_pred_scores = out.cpu().numpy()
    
    # 计算 ROC-AUC 分数
    return roc_auc_score(y_true, y_pred_scores)

# --- 4. 结果保存与绘图函数 ---
def plot_curves(results_dir, epochs, train_losses, valid_aucs):
    """绘制并保存训练曲线 (损失 vs ROC-AUC)"""
    os.makedirs(results_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True)

    # 修改: 绘制 ROC-AUC 曲线
    ax2.plot(epochs, valid_aucs, label='Validation ROC-AUC', color='tab:blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Validation ROC-AUC Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves_proteins600.png')
    plt.savefig(save_path)
    print(f"\n训练曲线图已保存到: {save_path}")

def save_summary(results_dir, best_valid_auc, test_auc):
    """将最终结果摘要保存到文本文件"""
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'proteins_results.txt')
    
    try:
        with open(summary_path, 'w') as f:
            f.write("MLP Model for ogbn-proteins\n")
            f.write("============================\n")
            f.write(f"Best Validation ROC-AUC: {best_valid_auc:.4f}\n")
            f.write(f"Test ROC-AUC on Best Model: {test_auc:.4f}\n")
        print(f"结果摘要已保存到: {summary_path}")
    except Exception as e:
        print(f"保存结果摘要时出错: {e}")

# --- 5. 主执行函数 ---
def main():
    # --- 参数设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/zhouruiqi/project/HW2/MLP/dataset/proteins'
    results_dir = '/home/zhouruiqi/project/HW2/MLP/results'
    model_results_dir = os.path.join(results_dir, 'model_weights')
    best_model_path = os.path.join(model_results_dir, 'proteins_mlp.pth')

    os.makedirs(model_results_dir, exist_ok=True)

    num_layers = 4      # 从 3 增加到 5
    hidden_channels = 256 # 从 256 增加到 512
    dropout = 0.5
    
    # 训练参数
    lr = 5e-3
    epochs = 600
    eval_steps = 10
    # --- 加载数据 ---
    loaded_data, num_tasks = load_data_from_csv(data_dir, device)
    if loaded_data is None:
        return

    # --- 初始化模型、优化器和损失函数 ---
    model = MLP(
        in_channels=loaded_data['x_train'].size(-1),
        hidden_channels=hidden_channels,
        out_channels=num_tasks, # 输出维度为112
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 修改: 使用 BCEWithLogitsLoss 应对多标签分类
    loss_fn = nn.BCEWithLogitsLoss()

    # --- 开始训练和评估循环 ---
    best_valid_auc = 0
    epoch_list, loss_list, auc_list = [], [], []

    print("\n--- 开始训练 ---")
    pbar = tqdm(range(1, epochs + 1), desc="Training Epochs")
    for epoch in pbar:
        loss = train(model, loaded_data['x_train'], loaded_data['y_train'], optimizer, loss_fn)
        
        if epoch % eval_steps == 0:
            valid_auc = evaluate(model, loaded_data['x_valid'], loaded_data['y_valid'])
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                torch.save(model.state_dict(), best_model_path)
            
            epoch_list.append(epoch)
            loss_list.append(loss)
            auc_list.append(valid_auc)
            
            pbar.set_postfix(loss=f'{loss:.4f}', valid_auc=f'{valid_auc:.4f}')

    pbar.close()
    print("\n--- 训练完成 ---")
    print(f"最终最佳验证集 ROC-AUC: {best_valid_auc:.4f}")

    # --- 在测试集上评估最佳模型 ---
    print("\n--- 开始在测试集上评估最佳模型 ---")
    best_model = MLP(
        in_channels=loaded_data['x_train'].size(-1),
        hidden_channels=hidden_channels,
        out_channels=num_tasks,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    test_auc = evaluate(best_model, loaded_data['x_test'], loaded_data['y_test'])
    print(f"最佳模型在测试集上的 ROC-AUC: {test_auc:.4f}")

    # --- 保存结果和绘图 ---
    plot_curves(results_dir, epoch_list, loss_list, auc_list)
    save_summary(results_dir, best_valid_auc, test_auc)

if __name__ == "__main__":
    main()