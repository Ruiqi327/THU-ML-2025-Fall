import pandas as pd
import json
import os
from tqdm import tqdm

def get_arxiv_category_map():
    """
    返回一个从 arXiv 代码到完整类别名称的映射字典。
    """
    return {
        'cs.AI': 'Artificial Intelligence',
        'cs.AR': 'Hardware Architecture',
        'cs.CC': 'Computational Complexity',
        'cs.CE': 'Computational Engineering, Finance, and Science',
        'cs.CG': 'Computational Geometry',
        'cs.CL': 'Computation and Language',
        'cs.CR': 'Cryptography and Security',
        'cs.CV': 'Computer Vision and Pattern Recognition',
        'cs.CY': 'Computers and Society',
        'cs.DB': 'Databases',
        'cs.DC': 'Distributed, Parallel, and Cluster Computing',
        'cs.DL': 'Digital Libraries',
        'cs.DM': 'Discrete Mathematics',
        'cs.DS': 'Data Structures and Algorithms',
        'cs.ET': 'Emerging Technologies',
        'cs.FL': 'Formal Languages and Automata Theory',
        'cs.GL': 'General Literature',
        'cs.GR': 'Graphics',
        'cs.GT': 'Computer Science and Game Theory',
        'cs.HC': 'Human-Computer Interaction',
        'cs.IR': 'Information Retrieval',
        'cs.IT': 'Information Theory', # 修正了之前的 's.IT'
        'cs.LG': 'Machine Learning',
        'cs.LO': 'Logic in Computer Science',
        'cs.MA': 'Multiagent Systems',
        'cs.MM': 'Multimedia',
        'cs.MS': 'Mathematical Software',
        'cs.NA': 'Numerical Analysis',
        'cs.NE': 'Neural and Evolutionary Computing',
        'cs.NI': 'Networking and Internet Architecture',
        'cs.OH': 'Other Computer Science',
        'cs.OS': 'Operating Systems',
        'cs.PF': 'Performance',
        'cs.PL': 'Programming Languages',
        'cs.RO': 'Robotics',
        'cs.SC': 'Symbolic Computation',
        'cs.SD': 'Sound',
        'cs.SE': 'Software Engineering',
        'cs.SI': 'Social and Information Networks',
        'cs.SY': 'Systems and Control',
    }

def create_test_subset_json(subset_size=500):
    """
    整合 ogbn-arxiv 数据，为指定集合的前 N 个样本创建一个 JSON 文件。
    (修正版：确保类别代码正确转换)
    """
    print("开始处理 ogbn-arxiv 数据集 (修正版：确保类别名称转换)...")

    # --- 1. 定义文件路径 ---
    base_dir = 'ogb_data/ogbn_arxiv/'
    titleabs_path = os.path.join(base_dir, 'raw', 'titleabs.tsv')
    nodeidx2paperid_path = os.path.join(base_dir, 'mapping', 'nodeidx2paperid.csv.gz')
    label_path = os.path.join(base_dir, 'raw', 'node-label.csv.gz')
    label_mapping_path = os.path.join(base_dir, 'mapping', 'labelidx2arxivcategeory.csv.gz')
    
    # 注意：你之前的文件路径指向了 train.csv.gz，我保留了这个设置
    split_idx_path = os.path.join(base_dir, 'split', 'time', 'test.csv.gz')
    
    output_dir = 'LLM_data'
    output_path = os.path.join(output_dir, f'arxiv_test.json') # 输出文件名
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. 加载数据 ---
    print("正在加载所有必需的文件...")
    try:
        titleabs_df = pd.read_csv(titleabs_path, sep='\t', header=None, names=['paper_id', 'title', 'abstract'])
        nodeidx2paperid_df = pd.read_csv(nodeidx2paperid_path, compression='gzip', header=None, names=['paper_id'])
        labels_df = pd.read_csv(label_path, compression='gzip', header=None, names=['label_id'])
        label_map_df = pd.read_csv(label_mapping_path, compression='gzip')
        split_idx_df = pd.read_csv(split_idx_path, compression='gzip', header=None, names=['node_id'])
    except FileNotFoundError as e:
        print(f"\n错误：文件未找到 -> {e.filename}")
        return

    # --- 3. 创建映射字典 (核心修正) ---
    print("正在创建映射字典...")
    paperid_to_text = {row.paper_id: (row.title, row.abstract) for row in titleabs_df.itertuples()}
    category_code_map = get_arxiv_category_map()
    
    label_to_category = {}
    for _, row in label_map_df.iterrows():
        label_id = row['label idx']
        code_str = row['arxiv category'] # e.g., "arxiv cs dm"
        
        # *** 修正逻辑在这里 ***
        parts = code_str.strip().split() # ['arxiv', 'cs', 'dm']
        if len(parts) == 3:
            # 构造成 'cs.DM' 的标准格式
            clean_code = f"{parts[1].lower()}.{parts[2].upper()}"
        else:
            clean_code = code_str # 如果格式不符，则保留原样

        # 从映射中获取完整名称
        full_name = category_code_map.get(clean_code, code_str)
        label_to_category[label_id] = full_name

    # --- 4. 选取并整合数据 ---
    node_indices = split_idx_df['node_id'].head(subset_size).tolist()
    print(f"已选取目标集合的前 {len(node_indices)} 个样本。")

    final_data = []
    print("正在整合数据...")
    for node_id in tqdm(node_indices, desc="Processing nodes"):
        paper_id = nodeidx2paperid_df.iloc[node_id]['paper_id']
        title, abstract = paperid_to_text.get(paper_id, ("Not Found", "Not Found"))
        label_id = labels_df.iloc[node_id]['label_id']
        category_name = label_to_category.get(label_id, "Unknown Category")
        
        entry = {
            "node_id": node_id,
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "label_id": int(label_id),
            "category_name": category_name # 现在这里应该是完整名称
        }
        final_data.append(entry)

    # --- 5. 写入 JSON 文件 ---
    print(f"正在将结果写入到文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print("\n处理完成！")
    print(f"成功生成文件: {output_path}")

if __name__ == '__main__':
    create_test_subset_json(subset_size=500)