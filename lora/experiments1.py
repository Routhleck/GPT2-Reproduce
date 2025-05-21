# -*- coding: utf-8 -*-
"""
LoRA超参数调优和分析脚本
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from tqdm import tqdm

# 为了确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 设置工作目录
working_dir = './'
output_directory = os.path.join(working_dir, "lora_tuning_results")
os.makedirs(output_directory, exist_ok=True)

# 定义超参数搜索空间
NUM_EPOCHS = 25  # 训练的轮数
r_values = [4, 8, 16, 32]  # LoRA的秩
alpha_values = [1, 4, 8]  # LoRA的alpha值
dropout_values = [0.0, 0.05, 0.1, 0.2]  # dropout率
target_modules_options = [
    ["c_attn"],                             # 只有注意力层
    ["c_attn", "c_proj"],                   # 注意力层和投影层
    ["c_attn", "c_proj", "c_fc", "c_ffn"]   # 注意力层、投影层和前馈层
]

# 评估提示
eval_prompts = [
    "I love this movie because",
    "This film was terrible due to",
    "The acting in this movie was",
    "The director's vision for this film",
    "The special effects in this movie"
]

def load_model_and_tokenizer():
    """加载基础模型和tokenizer"""
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    foundation_model = AutoModelForCausalLM.from_pretrained(model_name)
    return foundation_model, tokenizer

def prepare_dataset(tokenizer):
    """准备IMDB数据集"""
    dataset = "noob123/imdb_review_3000"
    data = load_dataset(dataset)
    data = data.map(lambda samples: tokenizer(samples['review']), batched=True)
    train_sample = data["train"].select(range(100))  # 使用更多样本以获得更好的结果
    train_sample = train_sample.remove_columns('sentiment')
    
    # 创建一个小的验证集
    val_sample = data["train"].select(range(100, 120))
    val_sample = val_sample.remove_columns('sentiment')
    
    return train_sample, val_sample

def get_outputs(model, inputs, tokenizer, max_new_tokens=100):
    """生成文本输出"""
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs

class LossCallback(Trainer):
    """用于记录训练损失的回调"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_loss = []
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """记录每个训练步骤的损失"""
        outputs = super().training_step(model, inputs, num_items_in_batch)
        self.training_loss.append(outputs.item())
        return outputs


def train_and_evaluate(foundation_model, tokenizer, train_dataset, val_dataset, lora_config, 
                       config_name, num_epochs=NUM_EPOCHS, learning_rate=3e-2):
    """训练并评估一个特定的LoRA配置"""
    # 创建LoRA模型
    peft_model = get_peft_model(foundation_model, lora_config)
    
    # 设置训练参数
    config_dir = os.path.join(output_directory, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=config_dir,
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"  # 禁用wandb等报告
    )
    
    # 创建带有损失记录的训练器
    trainer = LossCallback(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 训练模型
    train_result = trainer.train()
    
    # 保存模型
    peft_model_path = os.path.join(config_dir, "lora_model")
    trainer.model.save_pretrained(peft_model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.training_loss)
    plt.title(f"Training Loss - {config_name}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(config_dir, "training_loss.png"))
    plt.close()
    
    # 评估生成能力
    loaded_model = PeftModel.from_pretrained(foundation_model, peft_model_path, is_trainable=False)
    
    generation_results = {}
    for prompt in eval_prompts:
        input_text = tokenizer(prompt, return_tensors="pt")
        outputs = get_outputs(loaded_model, input_text, tokenizer, max_new_tokens=100)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generation_results[prompt] = generated_text
    
    # 保存生成结果
    with open(os.path.join(config_dir, "generation_results.txt"), "w") as f:
        for prompt, text in generation_results.items():
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {text}\n")
            f.write("-" * 80 + "\n")
    
    # 返回评估指标和损失历史
    return {
        "final_loss": train_result.training_loss,
        "loss_history": trainer.training_loss,
        "generation_results": generation_results
    }

def run_hyperparameter_search():
    """运行超参数搜索"""
    # 加载模型和数据
    foundation_model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    results = {}
    
    # 测试不同的r值
    for r in r_values:
        config_name = f"r_{r}"
        lora_config = LoraConfig(
            r=r,
            lora_alpha=1,  # 固定其他参数
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="lora_only",
            task_type="CAUSAL_LM"
        )
        print(f"Training with {config_name}")
        results[config_name] = train_and_evaluate(
            foundation_model, tokenizer, train_dataset, val_dataset, 
            lora_config, config_name
        )
    
    # 测试不同的alpha值
    for alpha in alpha_values:
        config_name = f"alpha_{alpha}"
        lora_config = LoraConfig(
            r=8,  # 固定r为中等值
            lora_alpha=alpha,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="lora_only",
            task_type="CAUSAL_LM"
        )
        print(f"Training with {config_name}")
        results[config_name] = train_and_evaluate(
            foundation_model, tokenizer, train_dataset, val_dataset, 
            lora_config, config_name
        )
    
    # 测试不同的dropout值
    for dropout in dropout_values:
        config_name = f"dropout_{dropout}"
        lora_config = LoraConfig(
            r=8,
            lora_alpha=1,
            target_modules=["c_attn"],
            lora_dropout=dropout,
            bias="lora_only",
            task_type="CAUSAL_LM"
        )
        print(f"Training with {config_name}")
        results[config_name] = train_and_evaluate(
            foundation_model, tokenizer, train_dataset, val_dataset, 
            lora_config, config_name
        )
    
    # 测试不同的目标模块
    for i, modules in enumerate(target_modules_options):
        config_name = f"modules_option_{i+1}"
        lora_config = LoraConfig(
            r=8,
            lora_alpha=1,
            target_modules=modules,
            lora_dropout=0.05,
            bias="lora_only",
            task_type="CAUSAL_LM"
        )
        print(f"Training with {config_name}")
        results[config_name] = train_and_evaluate(
            foundation_model, tokenizer, train_dataset, val_dataset, 
            lora_config, config_name
        )
    
    # 生成超参数比较报告
    generate_comparison_report(results)
    
    return results

def generate_comparison_report(results):
    """生成超参数比较报告"""
    # 创建比较图表
    plt.figure(figsize=(15, 10))
    
    # 绘制r值比较
    plt.subplot(2, 2, 1)
    r_configs = [f"r_{r}" for r in r_values]
    final_losses = [results[config]["final_loss"] for config in r_configs if config in results]
    plt.bar(r_configs, final_losses)
    plt.title("Final Loss by r value")
    plt.xticks(rotation=45)
    
    # 绘制alpha值比较
    plt.subplot(2, 2, 2)
    alpha_configs = [f"alpha_{alpha}" for alpha in alpha_values]
    final_losses = [results[config]["final_loss"] for config in alpha_configs if config in results]
    plt.bar(alpha_configs, final_losses)
    plt.title("Final Loss by alpha value")
    plt.xticks(rotation=45)
    
    # 绘制dropout值比较
    plt.subplot(2, 2, 3)
    dropout_configs = [f"dropout_{dropout}" for dropout in dropout_values]
    final_losses = [results[config]["final_loss"] for config in dropout_configs if config in results]
    plt.bar(dropout_configs, final_losses)
    plt.title("Final Loss by dropout value")
    plt.xticks(rotation=45)
    
    # 绘制目标模块比较
    plt.subplot(2, 2, 4)
    module_configs = [f"modules_option_{i+1}" for i in range(len(target_modules_options))]
    final_losses = [results[config]["final_loss"] for config in module_configs if config in results]
    plt.bar(module_configs, final_losses)
    plt.title("Final Loss by target modules")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "hyperparameter_comparison.png"))
    plt.close()
    
    # 创建文本报告
    with open(os.path.join(output_directory, "hyperparameter_analysis.txt"), "w") as f:
        f.write("# LoRA超参数调优分析报告\n\n")
        
        # 分析r值的影响
        f.write("## r值（秩）的影响\n\n")
        f.write("r值控制LoRA适配器的表达能力。更高的r值意味着更多的可训练参数，可能提供更好的适应性，但也增加了过拟合的风险。\n\n")
        for config in r_configs:
            if config in results:
                f.write(f"- {config}: 最终损失 = {results[config]['final_loss']:.4f}\n")
        f.write("\n")
        
        # 分析alpha值的影响
        f.write("## lora_alpha（缩放因子）的影响\n\n")
        f.write("alpha值控制了LoRA更新的缩放程度。较大的alpha值会放大LoRA的影响，可能导致更快的学习但也可能导致不稳定。\n\n")
        for config in alpha_configs:
            if config in results:
                f.write(f"- {config}: 最终损失 = {results[config]['final_loss']:.4f}\n")
        f.write("\n")
        
        # 分析dropout值的影响
        f.write("## lora_dropout的影响\n\n")
        f.write("dropout率影响模型的泛化能力。较高的dropout可以防止过拟合，但可能影响模型学习能力。\n\n")
        for config in dropout_configs:
            if config in results:
                f.write(f"- {config}: 最终损失 = {results[config]['final_loss']:.4f}\n")
        f.write("\n")
        
        # 分析目标模块的影响
        f.write("## target_modules的影响\n\n")
        f.write("不同的目标模块会影响LoRA适配哪些层。某些层可能对特定任务更重要。\n\n")
        for i, modules in enumerate(target_modules_options):
            config = f"modules_option_{i+1}"
            if config in results:
                f.write(f"- {config} ({modules}): 最终损失 = {results[config]['final_loss']:.4f}\n")
        f.write("\n")
        
        # 最优超参数组合分析
        f.write("## 最优超参数组合分析\n\n")
        all_configs = list(results.keys())
        best_config = min(all_configs, key=lambda x: results[x]["final_loss"] if "final_loss" in results[x] else float('inf'))
        f.write(f"根据实验结果，最优的超参数组合是 {best_config}，最终损失为 {results[best_config]['final_loss']:.4f}。\n\n")
        
        # 生成质量分析
        f.write("## 生成质量分析\n\n")
        f.write("不同超参数设置对生成质量的影响可以在各配置目录下的generation_results.txt文件中查看。\n")
        f.write("总体而言，较低的最终损失通常对应更好的生成质量，但这不是绝对的。某些配置可能在特定类型的提示上表现更好。\n")

def main():
    """主函数"""
    print("开始LoRA超参数调优...")
    results = run_hyperparameter_search()
    print(f"超参数调优完成。结果保存在 {output_directory} 目录下。")

if __name__ == "__main__":
    main()
