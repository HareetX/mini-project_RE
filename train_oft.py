import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import BOFTConfig, get_peft_model
from trl import SFTTrainer

from src.duie_dataset import read_dataset, format_example_wo_schema, format_example_w_schema

# 加载模型与分词器
model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 配置 OFT (Orthogonal Finetuning) 参数
boft_config = BOFTConfig(
    boft_block_size=4,               # 块大小，通常设置为 4 或 8
    boft_n_butterfly_factor=2,       # 蝴蝶因子层数，数值越大参数量越多，表达能力越强
    target_modules=[                 # 推荐覆盖 Qwen2.5 的所有线性层以获得最佳效果
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    boft_dropout=0.1,                # 防止过拟合的 Dropout
    bias="none",
    task_type="CAUSAL_LM"
)

# 将模型转换为 PEFT 模型
model = get_peft_model(model, boft_config)
model.print_trainable_parameters()

# 准备关系抽取数据集
train_data = read_dataset("data/train.json")

# 将数据转换为单轮对话的 Prompt 格式
def format_prompts(example):
    return format_example_wo_schema(example)

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_prompts)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./qwen-oft-relation-extraction",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,             # 密集记录日志以获取平滑的 Loss 曲线
    max_steps=50,                # 实际训练时建议使用 num_train_epochs=3
    save_strategy="steps",
    save_steps=10,
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,
    report_to="tensorboard"      # 启用 TensorBoard 以便导出 Loss 曲线图片
)

# 初始化 SFTTrainer 进行微调
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# 开始训练
print("开始训练...")
trainer.train()

# 保存微调后的权重
trainer.model.save_pretrained("./qwen-oft-final")
print("模型保存完毕！")