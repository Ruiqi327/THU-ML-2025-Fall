import os
from ogb.nodeproppred import NodePropPredDataset

def load_and_print_dataset(dataset_name, root_dir='./data'):
    """
    加载一个 OGB 节点属性预测数据集，获取其标准划分，并打印统计信息。

    Args:
        dataset_name (str): 要加载的数据集名称 (例如 'ogbn-products')。
        root_dir (str): 存储数据集的根目录。
    """
    print(f"--- 开始加载数据集: {dataset_name} ---")
    
    try:
        # 确保数据根目录存在
        os.makedirs(root_dir, exist_ok=True)
        
        # 下载并加载数据集
        # NodePropPredDataset 会自动处理下载和解压
        dataset = NodePropPredDataset(name=dataset_name, root=root_dir)
        
        print("数据集加载成功。")
        
        # 获取图对象和标签
        # 对于节点预测任务，通常只有一个图，所以我们使用 dataset[0]
        graph, labels = dataset[0]
        
        # 获取标准的数据划分
        # 返回一个包含 'train', 'valid', 'test' 键的字典
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]
        
        # 打印图的统计信息
        print("图信息:")
        print(f"  - 节点数量: {graph['num_nodes']:,}")
        print(f"  - 边数量: {graph['edge_index'].shape[1]:,}")
        
        # 打印数据划分的统计信息
        print("数据划分:")
        print(f"  - 训练集样本数: {len(train_idx):,}")
        print(f"  - 验证集样本数: {len(valid_idx):,}")
        print(f"  - 测试集样本数: {len(test_idx):,}")
        
        # 打印标签信息
        print(f"标签形状: {labels.shape}")
        print(f"数据集 '{dataset_name}' 的统计信息打印完毕。\n")

    except Exception as e:
        print(f"加载或处理数据集 {dataset_name} 时发生错误: {e}\n")

def main():
    # 需要加载的数据集列表
    datasets_to_load = ["ogbn-products", "ogbn-proteins", "ogbn-arxiv"]
    
    # 指定一个目录来存放所有下载的数据
    data_storage_path = './ogb_data'
    
    for name in datasets_to_load:
        load_and_print_dataset(name, root_dir=data_storage_path)

if __name__ == "__main__":
    # 确保你已经安装了 ogb 包:
    # pip install ogb
    main()