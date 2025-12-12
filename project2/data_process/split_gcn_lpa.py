import os
import gzip
import shutil
import pandas as pd

def decompress_gz_to_csv(source_path, destination_path):
    """
    解压一个 .gz 文件并将其内容保存为 .csv 文件。
    """
    # 确保目标目录存在
    dest_dir = os.path.dirname(destination_path)
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        with gzip.open(source_path, 'rb') as f_in:
            with open(destination_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"成功解压: {source_path} -> {destination_path}")
    except FileNotFoundError:
        print(f"错误：源文件未找到 -> {source_path}")
    except Exception as e:
        print(f"解压文件 {source_path} 时发生错误: {e}")

def main():
    # --- 路径配置 ---
    # OGB 原始数据所在的根目录
    ogb_raw_dir = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_products/raw'
    ogb_split_dir = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_products/split/sales_ranking'
    
    # GCN 模型所需的数据集输出目录
    output_dir = '/home/zhouruiqi/project/HW2/GCN/dataset/product'
    print(f"--- 开始为 GCN 准备标准转导式数据集 ---")
    print(f"--- 输出目录: {output_dir} ---")

    # --- 文件映射 ---
    # 定义需要处理的文件及其源路径和目标文件名
    files_to_process = {
        # 文件类型      源文件路径                                     目标文件名
        'node_feat':   (os.path.join(ogb_raw_dir, 'node-feat.csv.gz'),   'node_feat.csv'),
        'node_label':  (os.path.join(ogb_raw_dir, 'node-label.csv.gz'),  'node_label.csv'),
        'edge':        (os.path.join(ogb_raw_dir, 'edge.csv.gz'),        'edge.csv'),
        'train_idx':   (os.path.join(ogb_split_dir, 'train.csv.gz'),     'train_idx.csv'),
        'valid_idx':   (os.path.join(ogb_split_dir, 'valid.csv.gz'),     'valid_idx.csv'),
        'test_idx':    (os.path.join(ogb_split_dir, 'test.csv.gz'),      'test_idx.csv'),
    }

    # --- 执行解压和移动 ---
    for file_type, (source_path, dest_filename) in files_to_process.items():
        destination_path = os.path.join(output_dir, dest_filename)
        decompress_gz_to_csv(source_path, destination_path)

    print("\n--- 所有文件处理完毕 ---")
    print("现在您可以使用 GCN 训练脚本从本地文件加载数据了。")

if __name__ == "__main__":
    main()