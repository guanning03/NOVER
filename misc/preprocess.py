import random
import re
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from huggingface_hub import create_repo

# 过滤函数
def is_valid(sample):
    ans = sample.get("reference_answer", None)

    if not isinstance(ans, str):
        return False

    ans = ans.strip()

    if not ans:
        return False

    # 纯数字、标点等
    if re.fullmatch(r"[\d\W\s]+", ans):
        return False

    # 少于30字符
    if len(ans) < 30:
        return False

    # 太短（单词少）
    if len(ans.split()) <= 10:
        return False

    # 没有完整句号数量（防止句子不全）
    if ans.count('.') + ans.count('!') + ans.count('?') <= 2:
        return False

    return True

# 加载数据集
print("📦 Loading dataset...")
dataset = load_dataset("GeneralReasoning/GeneralThought-430K", split="train")

# 过滤样本
print("🔍 Filtering samples...")
filtered = [sample for sample in tqdm(dataset) if is_valid(sample)]
print(f"✅ Filtered samples: {len(filtered)}")

if len(filtered) < 5000:
    raise ValueError("❌ Too few samples after filtering (<5000).")

# 划分数据集
random.seed(42)
random.shuffle(filtered)

train_data = filtered[:3000]
val_data = filtered[3000:4000]
test_data = filtered[4000:5000]

# 构建 DatasetDict
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)
test_ds = Dataset.from_list(test_data)

final_ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

# 创建远程仓库
repo_id = "guanning-ai/GeneralThought-430K-filtered"
print(f"🚀 Creating repo: {repo_id}")
create_repo(repo_id, repo_type="dataset", exist_ok=True)

# 上传数据集到 hub
print("⬆️ Pushing dataset to the Hub...")
final_ds.push_to_hub(repo_id)
print("✅ Upload complete.")
