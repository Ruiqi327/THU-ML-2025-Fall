import json
import re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest # 导入 LoRARequest
from tqdm import tqdm

# --- 1. 配置参数 ---

# 基础模型的路径或HuggingFace名称
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  

ADAPTER_PATH = "/home/zhouruiqi/project/LLaMA-Factory/output/qwen2.5/arxiv/checkpoint-500"

# 输入与输出文件路径
TEST_FILE_PATH = "/home/zhouruiqi/project/HW2/LLM/LLM_data/arxiv_test.json"
RESULTS_FILE_PATH = "/home/zhouruiqi/project/HW2/LLM/inference_results_with_lora.json"

# 推理时使用的指令
INSTRUCTION = "Classify the following research paper into its appropriate category based on the title and abstract. The category name follows from standard arxiv disciplines (full name, no abbreviation. For example, Machine Learning, Computational Linguistics, etc). You should put the category in the block <answer></answer>."

# vLLM 配置
TENSOR_PARALLEL_SIZE = 1 
GPU_MEMORY_UTILIZATION = 0.9

# 采样参数
sampling_params = SamplingParams(temperature=0.9, max_tokens=512)


# --- 2. 数据加载与预处理 ---

def create_prompt(item):
    """根据测试数据项构建完整的模型输入提示。"""
    title = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    
    if not title or not abstract:
        return None
        
    input_text = f"Title: {title}\n Abstract: {abstract}\n"
    prompt = f"{INSTRUCTION}\n{input_text}"
    return prompt 

def load_test_data(filepath):
    """加载测试数据，返回原始数据列表、提示列表和真实标签列表。"""
    prompts = []
    ground_truths = []
    original_data = []
    print(f"正在从 {filepath} 加载测试数据...")
    with open(filepath, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        for item in tqdm(test_data, desc="预处理数据"):
            prompt = create_prompt(item)
            category = item.get("category_name", "").strip()
            if prompt and category:
                prompts.append(prompt)
                ground_truths.append(category)
                original_data.append(item)
    print(f"数据加载完成，共 {len(prompts)} 条有效测试样本。")
    return prompts, ground_truths, original_data

def extract_answer(text):
    """从模型输出中提取<answer>标签内的内容。"""
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        return match.group(1).strip()
    return ""


# --- 3. 模型加载与推理 ---

print("正在加载 vLLM 模型...")
llm = LLM(
    model=MODEL_PATH, 
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    enable_lora=True  # 启用LoRA支持
)
print(f"基础模型 '{MODEL_PATH}' 加载完成，LoRA已启用。")

# 加载数据
prompts, ground_truths, original_data = load_test_data(TEST_FILE_PATH)

print("开始批量推理（应用LoRA adapter）...")
# 使用 vLLM 进行批量推理，并指定要使用的LoRA adapter
outputs = llm.generate(
    prompts, 
    sampling_params,
    lora_request=LoRARequest(
        lora_name="my_adapter",  # 给你的adapter起一个名字
        lora_int_id=1,           # 给一个唯一的整数ID
        lora_local_path=ADAPTER_PATH
    )
)
print("推理完成。")


# --- 4. 评估、结果计算与保存 ---

correct_predictions = 0
total_predictions = len(outputs)
results_to_save = []

print("开始评估结果并保存...")
for i in tqdm(range(total_predictions), desc="评估与保存"):
    original_item = original_data[i]
    generated_text = outputs[i].outputs[0].text
    
    predicted_category = extract_answer(generated_text)
    true_category = ground_truths[i]
    
    normalized_predicted = predicted_category.lower().strip()
    normalized_true = true_category.lower().strip()
    
    is_correct = (normalized_predicted == normalized_true)
    
    if is_correct:
        correct_predictions += 1
        
    result_item = {
        "title": original_item.get("title"),
        "abstract": original_item.get("abstract"),
        "true_category": true_category,
        "predicted_category": predicted_category,
        "model_raw_output": generated_text.strip(),
        "is_correct": is_correct
    }
    results_to_save.append(result_item)

# --- 5. 最终结果输出与文件写入 ---

accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

with open(RESULTS_FILE_PATH, "w", encoding="utf-8") as f_out:
    json.dump(results_to_save, f_out, ensure_ascii=False, indent=4)
print(f"\n详细推理结果已保存至: {RESULTS_FILE_PATH}")

print("\n--- 推理与评估摘要 ---")
print(f"总样本数: {total_predictions}")
print(f"正确预测数: {correct_predictions}")
print(f"模型正确率: {accuracy:.2f}%")

print("\n--- 随机错误样本对比 ---")
error_samples = [res for res in results_to_save if not res['is_correct']]
for i, sample in enumerate(error_samples[:5]):
    print(f"错误样本 {i+1}:")
    print(f"  真实标签: {sample['true_category']}")
    print(f"  预测标签: {sample['predicted_category']}")
    print("-" * 20)