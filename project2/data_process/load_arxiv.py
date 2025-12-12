from ogb.nodeproppred import NodePropPredDataset
import os

# 指定你存放所有 OGB 数据集的根目录
data_root = 'ogb_data'
dataset_name = 'ogbn-arxiv'

print(f"正在检查并下载 '{dataset_name}' 数据集到 '{data_root}' 目录...")
print("如果数据已存在，将跳过下载。")

# 这行代码是关键
# 它会自动检查 'ogb_data/ogbn-arxiv/' 目录
# 如果目录或其中的文件不完整，它会自动下载
dataset = NodePropPredDataset(name=dataset_name, root=data_root)

print("\n检查/下载完成！")

# 验证一下我们关心的文件是否存在
raw_file_path = os.path.join(data_root, dataset_name, 'raw', 'titleabs.tsv')
print(f"摘要文件路径: {raw_file_path}")

if os.path.exists(raw_file_path):
    print("成功找到摘要文件 (titleabs.tsv)！")
else:
    print("错误：未找到摘要文件，请检查下载过程是否有误。")
