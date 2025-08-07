import os
import json
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, HfArgumentParser
)
from peft import LoraConfig, get_peft_model
import swanlab
from dataclasses import dataclass, field

@dataclass
class DistillationArguments:
    teacher_model_path: str = field(metadata={"help": "教师模型的路径"})
    student_model_path: str = field(metadata={"help": "学生模型的路径"})
    dataset_path: str = field(default="data/dirty_chinese_dpo.json", metadata={"help": "用于蒸馏的数据集路径"})
    system_prompt: str = field(default="你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题。", metadata={"help": "系统提示语"})
    # 将蒸馏超参数添加到TrainingArguments中，以便DistillTrainer可以访问
    # output_dir: str = field(default="./output/distilled_adapter", metadata={"help": "蒸馏后学生模型适配器的保存目录"})
    
    # 蒸馏超参数
    temperature: float = field(default=2.0, metadata={"help": "蒸馏温度，用于平滑教师模型的输出概率"})
    alpha: float = field(default=0.5, metadata={"help": "蒸馏损失(KL散度)与SFT损失(交叉熵)的权重"})

    # LoRA 和训练配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})
    max_length: int = field(default=1024, metadata={"help": "输入最大长度"})
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab"})

class DistillTrainer(Trainer):
    def __init__(self, *args, teacher_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature # 从TrainingArguments获取
        self.alpha = self.args.alpha # 从TrainingArguments获取

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            teacher_logits = teacher_outputs.logits
        
        labels_mask = inputs["labels"].ne(-100)
        
        student_log_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        
        valid_student_log_probs = student_log_probs[labels_mask]
        valid_teacher_probs = teacher_probs[labels_mask]
        
        kd_loss = self.kl_loss_fct(valid_student_log_probs, valid_teacher_probs) * (self.temperature ** 2)
        
        loss = self.alpha * kd_loss + (1 - self.alpha) * student_loss
        
        return (loss, student_outputs) if return_outputs else loss

@dataclass
class CustomTrainingArguments(TrainingArguments):
    # 将蒸馏超参数添加到TrainingArguments中，以便DistillTrainer可以访问
    output_dir: str = field(default="./output/distilled_adapter", metadata={"help": "蒸馏后学生模型适配器的保存目录"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "梯度累积步数"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "梯度累积步数"})
    # 蒸馏超参数
    # temperature: float = field(default=2.0, metadata={"help": "蒸馏温度，用于平滑教师模型的输出概率"})
    # alpha: float = field(default=0.5, metadata={"help": "蒸馏损失(KL散度)与SFT损失(交叉熵)的权重"})


def main():
    # --- 1. 解析参数 ---
    parser = HfArgumentParser((DistillationArguments, CustomTrainingArguments))
    distill_args, training_args = parser.parse_args_into_dataclasses()

    # --- 路径检查 ---
    for path in [distill_args.teacher_model_path, distill_args.student_model_path, distill_args.dataset_path]:
        if not os.path.exists(path):
            print(f"❌错误: 输入路径 '{path}' 不存在。请检查配置。")
            exit()
    
    # 将蒸馏参数传递给训练参数
    training_args.temperature = distill_args.temperature
    training_args.alpha = distill_args.alpha
    training_args.report_to = "swanlab" if distill_args.use_swanlab else "none"

    # --- 2. 配置SwanLab ---
    if distill_args.use_swanlab:
        os.environ["SWANLAB_PROJECT"] = "qwen3-distill-foul-mouthed"
        swanlab.init(project="qwen3-distill-foul-mouthed", config={**vars(distill_args), **training_args.to_dict()})

    # --- 3. 加载数据集和Tokenizer ---
    print("🚀 3. 加载数据集和Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(distill_args.student_model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(distill_args.dataset_path, 'r', encoding='utf-8') as f: data = json.load(f)
    formatted_data = []
    for item in data:
        if 'conversations' in item and 'chosen' in item:
            human_input = "".join([turn['value'] for turn in item['conversations'] if turn.get('from') == 'human']).strip()
            chosen_response = item['chosen'].get('value', '')
            if human_input and chosen_response:
                formatted_data.append({"instruction": distill_args.system_prompt, "input": human_input, "output": chosen_response})
    
    dataset = Dataset.from_list(formatted_data)

    def process_func(example):
        instruction_part = tokenizer(f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        response_part = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)
        input_ids = instruction_part["input_ids"] + response_part["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction_part["attention_mask"] + response_part["attention_mask"] + [1]
        labels = [-100] * len(instruction_part["input_ids"]) + response_part["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) > distill_args.max_length:
            input_ids, attention_mask, labels = input_ids[:distill_args.max_length], attention_mask[:distill_args.max_length], labels[:distill_args.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

    # --- 4. 加载模型 ---
    print("📚 4. 正在加载教师模型...")
    teacher_model = AutoModelForCausalLM.from_pretrained(distill_args.teacher_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    print("🧑‍🎓 正在加载学生模型...")
    student_model = AutoModelForCausalLM.from_pretrained(distill_args.student_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    student_model.enable_input_require_grads()
    student_model.config.use_cache = False

    # --- 5. 配置LoRA ---
    print("🛠️ 5. 为学生模型配置LoRA...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=distill_args.lora_r, lora_alpha=distill_args.lora_alpha, lora_dropout=distill_args.lora_dropout,
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    
    # --- 6. 开始训练 ---
    print("🚀 6. 初始化并启动DistillTrainer...")
    trainer = DistillTrainer(
        model=student_model, teacher_model=teacher_model,
        args=training_args, train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    # --- 7. 保存模型 ---
    print(f"💾 7. 保存蒸馏后的学生模型适配器到: {training_args.output_dir}")
    os.makedirs(training_args.output_dir, exist_ok=True)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("✅ 蒸馏训练完成！")
    if distill_args.use_swanlab: swanlab.finish()

if __name__ == "__main__":
    main()