import pandas as pd

# 定义文件路径
file_path = '/home/zhouruiqi/project/HW2/ogb_data/ogbn_arxiv/raw/edge.csv.gz'

try:
    # 使用 pandas 读取 .gz 文件的前5行
    # header=None 表示文件没有标题行
    # nrows=5 表示只读取前5行数据
    # compression='gzip' 告诉 pandas 这是一个 gzip 压缩文件
    edge_head_df = pd.read_csv(
        file_path, 
        header=None, 
        nrows=5, 
        compression='gzip'
    )
    # 打印 DataFrame
    print(f"文件 '{file_path}' 的前五行内容：")
    print(edge_head_df)

except FileNotFoundError:
    print(f"错误：文件未找到 -> {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")