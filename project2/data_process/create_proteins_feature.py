import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def extract_indices_from_split_files(split_dir):
    """
    从指定的目录中读取 train.csv.gz, valid.csv.gz, 和 test.csv.gz 文件，
    并提取其中的节点编号。 (此函数保持不变)
    """
    if not os.path.isdir(split_dir):
        print(f"错误：索引目录不存在 -> {split_dir}")
        return None

    split_indices = {}
    file_names = {
        "train": "train.csv.gz",
        "valid": "valid.csv.gz",
        "test": "test.csv.gz"
    }

    print(f"--- 开始从 '{split_dir}' 提取索引 ---")
    for split_name, file_name in file_names.items():
        file_path = os.path.join(split_dir, file_name)
        if not os.path.exists(file_path):
            print(f"警告：文件 '{file_path}' 未找到，无法进行分割。")
            return None
        try:
            indices = pd.read_csv(file_path, compression='gzip', header=None).values.flatten()
            split_indices[split_name] = indices
            print(f"成功提取 '{split_name}' 集的 {len(indices):,} 个索引。")
        except Exception as e:
            print(f"读取索引文件 '{file_path}' 时出错: {e}")
            return None
    return split_indices

def compute_aggregated_edge_features(edge_path, edge_feat_path, num_nodes):
    """
    计算每个节点的入边特征平均值作为新的节点特征。

    Args:
        edge_path (str): 边列表文件路径。
        edge_feat_path (str): 边特征文件路径。
        num_nodes (int): 图中的总节点数。

    Returns:
        np.ndarray: 一个 [num_nodes, edge_feature_dim] 的矩阵，包含新的节点特征。
    """
    print("\n--- 开始计算基于入边聚合的新节点特征 ---")
    
    # 1. 加载边和边特征数据
    print("正在加载边和边特征...")
    try:
        edges = pd.read_csv(edge_path, compression='gzip', header=None).values
        edge_features = pd.read_csv(edge_feat_path, compression='gzip', header=None).values
    except Exception as e:
        print(f"加载边或边特征文件时出错: {e}")
        return None
    
    edge_feature_dim = edge_features.shape[1]
    print(f"加载完成。边数: {len(edges):,}, 边特征维度: {edge_feature_dim}")

    # 2. 创建一个邻接列表，用于存储每个节点的入边特征
    #    键是目标节点ID，值是其入边特征的列表
    incoming_edge_features = [[] for _ in range(num_nodes)]
    
    print("正在构建入边特征邻接表...")
    # edges[:, 1] 是所有边的目标节点
    # edges[:, 0] 是所有边的源节点
    # 我们关心的是指向目标节点的边
    for i in tqdm(range(len(edges)), desc="Processing Edges"):
        target_node = edges[i, 1]
        feature = edge_features[i]
        incoming_edge_features[target_node].append(feature)

    # 3. 计算每个节点的平均入边特征
    new_node_features = np.zeros((num_nodes, edge_feature_dim), dtype=np.float32)
    
    print("正在计算每个节点的平均特征...")
    for i in tqdm(range(num_nodes), desc="Aggregating Features"):
        if incoming_edge_features[i]: # 如果节点有入边
            # 计算特征列表的平均值
            new_node_features[i] = np.mean(incoming_edge_features[i], axis=0)
        # 如果没有入边，特征将保持为全零向量，这是默认值

    print("新节点特征计算完成。")
    return new_node_features

def save_new_features(new_features, split_indices, output_dir):
    """
    根据划分索引，保存新的节点特征。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- 准备将新特征保存到 '{output_dir}' ---")

    # 将 numpy 数组转换为 pandas DataFrame 以便使用 .iloc
    features_df = pd.DataFrame(new_features)

    for split_name, indices in split_indices.items():
        print(f"正在处理和保存 '{split_name}' 数据集...")
        
        # 使用 .iloc 按行号（即节点索引）选取数据
        split_df = features_df.iloc[indices]
        
        # 定义输出文件路径
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        
        try:
            split_df.to_csv(output_path, index=False, header=False)
            print(f"'{split_name}' 数据集已成功保存到: {output_path}")
            print(f"  - 保存的样本数: {len(split_df):,}")
        except Exception as e:
            print(f"保存文件 '{output_path}' 时出错: {e}")

def main():
    # --- 路径配置 ---
    raw_data_dir = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_proteins/raw'
    split_dir = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_proteins/split/species'
    output_dir = '/home/zhouruiqi/project/HW2/MLP/dataset/proteins'
    
    edge_path = os.path.join(raw_data_dir, 'edge.csv.gz')
    edge_feat_path = os.path.join(raw_data_dir, 'edge-feat.csv.gz')
    
    # ogbn-proteins 数据集的总节点数是固定的
    NUM_NODES = 132534

    # --- 执行步骤 ---
    # 1. 提取 train/valid/test 节点的索引
    split_indices = extract_indices_from_split_files(split_dir)
    if not split_indices:
        return

    # 2. 计算基于入边聚合的新节点特征
    new_node_features = compute_aggregated_edge_features(edge_path, edge_feat_path, NUM_NODES)
    if new_node_features is None:
        return

    # 3. 根据划分索引保存新的特征文件
    save_new_features(new_node_features, split_indices, output_dir)
    
    print("\n--- 所有操作完成 ---")

if __name__ == "__main__":
    # 确保你已经安装了 pandas, numpy, tqdm:
    # pip install pandas numpy tqdm
    main()