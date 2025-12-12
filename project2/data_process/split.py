import pandas as pd
import os
import numpy as np

def extract_indices_from_split_files(split_dir):
    """
    从指定的目录中读取 train.csv.gz, valid.csv.gz, 和 test.csv.gz 文件，
    并提取其中的节点编号。

    Args:
        split_dir (str): 包含划分文件的目录路径。
    
    Returns:
        dict: 一个字典，包含 'train', 'valid', 'test' 的节点索引 numpy 数组。
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
            # 直接读取 .gz 文件，header=None 表示文件没有标题行
            indices = pd.read_csv(file_path, compression='gzip', header=None).values.flatten()
            split_indices[split_name] = indices
            print(f"成功提取 '{split_name}' 集的 {len(indices):,} 个索引。")
            
        except Exception as e:
            print(f"读取索引文件 '{file_path}' 时出错: {e}")
            return None

    return split_indices

def split_and_save_features(node_feat_path, split_indices, output_dir):
    """
    根据提供的索引，分割节点特征文件并保存。

    Args:
        node_feat_path (str): 原始节点特征文件路径 (node-feat.csv.gz)。
        split_indices (dict): 包含 'train', 'valid', 'test' 索引的字典。
        output_dir (str): 保存分割后文件的目标目录。
    """
    if not os.path.exists(node_feat_path):
        print(f"错误：节点特征文件不存在 -> {node_feat_path}")
        return

    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- 准备将数据集保存到 '{output_dir}' ---")

    # 2. 加载节点特征数据
    print(f"正在加载节点特征文件: {node_feat_path} ...")
    try:
        # 假设特征文件没有表头
        node_features_df = pd.read_csv(node_feat_path, compression='gzip', header=None)
        print(f"节点特征加载完毕，总共有 {len(node_features_df):,} 个节点。")
    except Exception as e:
        print(f"加载节点特征文件时出错: {e}")
        return

    # 3. 根据索引进行分割并保存
    for split_name, indices in split_indices.items():
        print(f"正在处理 '{split_name}' 数据集...")
        
        # 使用 .iloc 按行号（即节点索引）选取数据
        split_df = node_features_df.iloc[indices]
        
        # 定义输出文件路径（保存为非压缩的 .csv）
        output_path = os.path.join(output_dir, f"{split_name}_label.csv")
        
        try:
            # 保存为 CSV 文件，不包含 pandas 的索引和表头
            split_df.to_csv(output_path, index=False, header=False)
            print(f"'{split_name}' 数据集已成功保存到: {output_path}")
            print(f"  - 保存的样本数: {len(split_df):,}")
        except Exception as e:
            print(f"保存文件 '{output_path}' 时出错: {e}")

def main():
    # --- 路径配置 ---
    # 包含 train.csv.gz, valid.csv.gz, test.csv.gz 的目录
    split_dir = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_arxiv/split/time'
    
    # 原始节点特征文件
    node_feat_path = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_arxiv/raw/node-label.csv.gz'
    
    # 分割后文件的输出目录
    output_dir = '/home/zhouruiqi/project/HW2/MLP/dataset/arxiv'

    # --- 执行步骤 ---
    # 1. 提取索引
    split_indices = extract_indices_from_split_files(split_dir)
    
    # 2. 如果成功提取索引，则进行分割和保存
    if split_indices:
        split_and_save_features(node_feat_path, split_indices, output_dir)
        print("\n--- 所有操作完成 ---")

if __name__ == "__main__":
    # 确保你已经安装了 pandas:
    # pip install pandas
    main()