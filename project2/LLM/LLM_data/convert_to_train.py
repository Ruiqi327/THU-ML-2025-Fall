import json

INSTRUCTION = "Classify the following research paper into its appropriate category based on the title and abstract. The category name follows from standard arxiv disciplines (full name, no abbreviation. For example, Machine Learning, Computational Linguistics, etc).  You should put the category in the block <answer></answer>, each paper has only one category."

# 创建一个空列表来存储所有转换后的对象
all_converted_data = []

with open("/home/zhouruiqi/project/HW2/LLM/LLM_data/arxiv_train.json", "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
    for item in data:
        title = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()
        categories = item.get("category_name", "").strip()
        
        if title and abstract and categories:
            new_obj = {
                "instruction": INSTRUCTION,
                "input": f"Title: {title}\n Abstract: {abstract}\n",
                "output": f"<answer>{categories}</answer>"
            }
            # 将新对象追加到列表中，而不是直接写入文件
            all_converted_data.append(new_obj)

# 循环结束后，将整个列表作为单个JSON对象写入文件
with open("/home/zhouruiqi/project/HW2/LLM/LLM_data/arxiv_train_converted.json", "w", encoding="utf-8") as f_out:
    # 使用 json.dump() 将列表写入文件，并使用 indent 参数进行格式化，使其更易读
    json.dump(all_converted_data, f_out, ensure_ascii=False, indent=4)

print("转换完成，已保存为标准的JSON格式文件。")