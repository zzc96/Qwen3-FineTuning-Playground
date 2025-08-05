import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, 
    DataCollatorWithPadding, HfArgumentParser
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import PPOConfig, PPOTrainer
import swanlab
from dataclasses import dataclass, field

@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "SFT合并后的模型路径"})
    rm_adapter_path: str = field(metadata={"help": "RM LoRA适配器路径"})
    dataset_path: str = field(default="data/dirty_chinese_dpo.json", metadata={"help": "数据集路径"})
    ppo_adapter_output_dir: str = field(default="./output/ppo_adapter", metadata={"help": "PPO LoRA适配器保存目录"})
    system_prompt: str = field(default="你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。", metadata={"help": "系统提示语"})
    
    # LoRA配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})

    # PPO配置
    learning_rate: float = field(default=1e-5, metadata={"help": "PPO学习率"})
    kl_coef: float = field(default=0.2, metadata={"help": "KL散度惩罚系数"})
    max_prompt_length: int = field(default=512, metadata={"help": "最大提示长度"})

    # 设备配置
    policy_device: str = field(default="cuda:0", metadata={"help": "策略模型和参考模型所在的设备"})
    reward_device: str = field(default="cuda:1", metadata={"help": "奖励模型和价值模型所在的设备"})
    
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab"})

def setup_swanlab(args: ScriptArguments):
    if not args.use_swanlab:
        return
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-rm-ppo-chinese"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    swanlab.init(
        project="qwen3-sft-rm-ppo-chinese",
        run_name="ppo-training-professional",
        config=vars(args)
    )

def load_prompts(dataset_path, tokenizer, system_prompt):
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌错误: 数据集文件未找到 at {dataset_path}")
        exit()
    
    prompts = []
    for item in data:
        if 'conversations' in item:
            human_input = "".join([turn['value'] + "\n" for turn in item['conversations'] if turn.get('from') == 'human']).strip()
            if human_input:
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": human_input}],
                    tokenize=False, add_generation_prompt=True
                )
                prompts.append({"query": formatted_prompt})
    return prompts

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # --- 路径检查 ---
    for path in [args.model_path, args.rm_adapter_path, args.dataset_path]:
        if not os.path.exists(path):
            print(f"❌错误: 输入路径 '{path}' 不存在。请检查配置。")
            exit()
            
    print("🚀 1. 配置和初始化 SwanLab...")
    setup_swanlab(args)

    print("🚀 2. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 3. 加载和预处理数据集...")
    all_prompts = load_prompts(args.dataset_path, tokenizer, args.system_prompt)
    full_dataset = Dataset.from_list(all_prompts)
    
    # 使用 train_test_split 划分数据集
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"PPO训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    def tokenize_fn(examples):
        return tokenizer(examples["query"], truncation=True, max_length=args.max_prompt_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=False)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_dataset = eval_dataset.map(tokenize_fn, batched=False)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print("🚀 4. 配置PPO...")
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate, report_to="swanlab" if args.use_swanlab else "none",
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_ppo_epochs=4, output_dir="./output/ppo_model_temp",
        num_train_epochs=1, gradient_checkpointing=True, kl_coef=args.kl_coef,
    )

    print("🚀 5. 创建策略模型 (Policy Model)...")
    ppo_lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.policy_device
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    model = get_peft_model(model, ppo_lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("🚀 6. 创建奖励模型和价值模型...")
    # 奖励模型
    rm_model_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.reward_device
    )
    rm_model_base.config.pad_token_id = tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(rm_model_base, args.rm_adapter_path)
    reward_model.eval()
    print("奖励模型加载完成。")
    
    # 价值模型 (带LoRA)
    value_lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.reward_device
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    value_model = get_peft_model(value_model, value_lora_config)
    print("价值模型可训练参数:")
    value_model.print_trainable_parameters()
    
    print("🚀 7. 创建并启动PPOTrainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config, model=model, ref_model=None, reward_model=reward_model, value_model=value_model,
        processing_class=tokenizer, train_dataset=train_dataset, data_collator=DataCollatorWithPadding(tokenizer),
        eval_dataset=eval_dataset,
    )
    ppo_trainer.train()

    print(f"💾 8. 保存PPO LoRA适配器到: {args.ppo_adapter_output_dir}")
    os.makedirs(args.ppo_adapter_output_dir, exist_ok=True)
    ppo_trainer.save_model(args.ppo_adapter_output_dir)
    
    print("✅ PPO训练完成!")
    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    main()
    