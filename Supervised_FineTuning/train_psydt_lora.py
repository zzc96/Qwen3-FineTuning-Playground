# LLM_fine_turning/train_psydt_lora.py
# YIRONGCHEN/PsyDTCorpus/train_psydt_lora.py
import os
import json
import argparse
from typing import List, Dict
from inspect import signature

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import snapshot_download
import swanlab


def build_samples_from_messages(messages: List[Dict], tokenizer, max_length: int):
    """
    使用 Qwen3 对话模板显式构造样本：
    <|im_start|>system ... <|im_end|>
    <|im_start|>user ... <|im_end|>
    <|im_start|>assistant ... <|im_end|>
    规则：每个 assistant 回合生成 1 个样本；仅对该回合的回复位置计算 loss，其余为 -100。
    """
    def render_block(role: str, content: str) -> str:
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # 拆出可选的 system
    system_msgs = [m for m in messages if m.get("role") == "system"]
    system_content = system_msgs[0]["content"] if len(system_msgs) > 0 else None

    # 仅保留 user/assistant，按顺序
    conv = []
    if system_content:
        conv.append({"role": "system", "content": system_content})
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            conv.append({"role": m["role"], "content": m["content"]})

    samples = []

    # 遍历每个 assistant 回合
    for i, msg in enumerate(conv):
        if msg["role"] != "assistant":
            continue

        # 上下文（到本轮 assistant 前）
        context_msgs = conv[:i]

        # 显式拼装 instruction（上下文完整块 + 本轮 assistant 头，不含回复）
        prefix_text = ""
        for cm in context_msgs:
            prefix_text += render_block(cm["role"], cm["content"])
        instruction_text = prefix_text + "<|im_start|>assistant\n"  # 不加 <|im_end|>

        # 回复部分（带结尾 <|im_end|>）
        resp_text = f"{msg['content']}<|im_end|>"

        # 分词
        instruction_part = tokenizer(instruction_text, add_special_tokens=False)
        response_part = tokenizer(resp_text, add_special_tokens=False)

        input_ids = instruction_part["input_ids"] + response_part["input_ids"]
        attention_mask = instruction_part["attention_mask"] + response_part["attention_mask"]
        labels = [-100] * len(instruction_part["input_ids"]) + response_part["input_ids"]

        # 长度截断（左截断，尽量保留结尾与完整回答）
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            labels = labels[-max_length:]

        samples.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        )

    return samples


def load_psydt_dataset(train_path: str, eval_path: str, tokenizer, max_length: int, max_train_items: int = None, max_eval_items: int = None):
    """
    读取 PsyDTCorpus 的 JSON 文件（数组），展开为样本列表：
    - 每条对话 -> 多个样本（每个 assistant 回合一个样本）
    """
    def load_file(path, max_items=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # 文件是一个 JSON 数组
        if max_items is not None:
            data = data[:max_items]
        all_samples = []
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue
            samples = build_samples_from_messages(messages, tokenizer, max_length)
            all_samples.extend(samples)
        return all_samples

    train_samples = load_file(train_path, max_items=max_train_items)
    eval_samples = load_file(eval_path, max_items=max_eval_items) if eval_path and os.path.exists(eval_path) else []

    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples) if len(eval_samples) > 0 else None
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser()
    # 模型与数据集参数
    parser.add_argument("--model_repo", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_local_dir", default="./Qwen/Qwen3-1.7B")

    # 训练/评估文件名（若不存在将自动下载）
    parser.add_argument("--train_file", default="./PsyDTCorpus_train_mulit_turn_packing.json")
    parser.add_argument("--eval_file", default="./PsyDTCorpus_test_single_turn_split.json")

    # 自动下载数据集所需参数
    parser.add_argument("--dataset_repo", default="YIRONGCHEN/PsyDTCorpus")
    parser.add_argument("--dataset_dir", default="./data/PsyDTCorpus")

    # 训练超参
    parser.add_argument("--output_dir", default="./output/qwen3-1_7b-psydt-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)

    # 开启评估，并限制数据量：训练集300条对话，测试集20条对话
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--max_train_items", type=int, default=300, help="限制读取训练对话条数（默认300）")
    parser.add_argument("--max_eval_items", type=int, default=20, help="限制读取评估对话条数（默认20）")

    # SwanLab 相关
    parser.add_argument("--swanlab_project", default="qwen3-psydt-lora", help="SwanLab 项目名")
    parser.add_argument("--swanlab_run_name", default="qwen3-1_7b-psydt-lora", help="SwanLab 运行名")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # SwanLab 环境与配置（与 HF-Trainer 集成）
    os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    swanlab.config.update({
        "model": args.model_repo,
        "max_length": args.max_length,
        "lora_r": 8,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_file": args.train_file,
        "eval_file": args.eval_file if args.do_eval else "",
        "max_train_items": args.max_train_items,
        "max_eval_items": args.max_eval_items
    })

    # 自动下载模型（若本地目录不存在则下载，并使用返回路径）
    if not os.path.exists(args.model_local_dir):
        print(">>> 模型本地目录不存在，开始下载模型...")
        model_path = snapshot_download(args.model_repo, cache_dir="./", revision="master")
    else:
        model_path = args.model_local_dir

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # LoRA 配置（r=8）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # 自动下载数据集（若本地缺失）
    train_name = os.path.basename(args.train_file) if args.train_file else "PsyDTCorpus_train_mulit_turn_packing.json"
    eval_name = os.path.basename(args.eval_file) if (args.eval_file and args.do_eval) else "PsyDTCorpus_test_single_turn_split.json"
    need_download = (not os.path.exists(args.train_file)) or (args.do_eval and args.eval_file and not os.path.exists(args.eval_file))

    if need_download:
        print(">>> 自动下载数据集文件...")
        os.makedirs(args.dataset_dir, exist_ok=True)
        try:
            ds_path = snapshot_download(
                args.dataset_repo,
                repo_type="dataset",
                cache_dir=args.dataset_dir,
                allow_patterns=[train_name, eval_name] if args.do_eval else [train_name]
            )
        except TypeError:
            # 兼容部分 modelscope 版本的参数名差异
            ds_path = snapshot_download(
                args.dataset_repo,
                cache_dir=args.dataset_dir,
            )

        # 尝试在返回目录下定位文件
        def locate_file(root_dir, fname):
            cand = os.path.join(root_dir, fname)
            if os.path.exists(cand):
                return cand
            for r, _, files in os.walk(root_dir):
                if fname in files:
                    return os.path.join(r, fname)
            return None

        train_candidate = locate_file(ds_path, train_name)
        if train_candidate:
            args.train_file = train_candidate
        else:
            raise FileNotFoundError(f"未找到训练文件 {train_name}（下载目录：{ds_path}）")

        if args.do_eval:
            eval_candidate = locate_file(ds_path, eval_name)
            if eval_candidate:
                args.eval_file = eval_candidate
            else:
                print(f">>> 未在下载目录中找到评估文件 {eval_name}，将不执行评估。")
                args.do_eval = False

        print(f">>> 使用训练文件: {args.train_file}")
        if args.do_eval:
            print(f">>> 使用评估文件: {args.eval_file}")

    # 加载数据（限制条数：训练300，对应 --max_train_items；评估20，对应 --max_eval_items）
    print(">>> 开始读取数据集...")
    train_dataset, eval_dataset = load_psydt_dataset(
        args.train_file,
        args.eval_file if args.do_eval else "",
        tokenizer,
        max_length=args.max_length,
        max_train_items=args.max_train_items,
        max_eval_items=args.max_eval_items,
    )
    print(f">>> 训练样本数: {len(train_dataset)}",
          f"评估样本数: {len(eval_dataset) if eval_dataset else 0}")

    # 训练参数（启用 SwanLab，兼容不同 transformers 版本）
    strat_val = "steps" if (args.do_eval and eval_dataset is not None) else "no"
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
    )
    # 新增：每个 epoch 结束保存一次 checkpoint
    kwargs["save_strategy"] = "epoch"
    ta_sig = signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        kwargs["evaluation_strategy"] = "epoch" if (args.do_eval and eval_dataset is not None) else "no"
    elif "eval_strategy" in ta_sig.parameters:
        kwargs["eval_strategy"] = "epoch" if (args.do_eval and eval_dataset is not None) else "no"
    if "save_on_each_node" in ta_sig.parameters:
        kwargs["save_on_each_node"] = True
    if "bf16" in ta_sig.parameters and torch.cuda.is_available():
        kwargs["bf16"] = True
    if "report_to" in ta_sig.parameters:
        kwargs["report_to"] = "swanlab"
    if "run_name" in ta_sig.parameters:
        kwargs["run_name"] = args.swanlab_run_name

    training_args = TrainingArguments(**kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if (args.do_eval and eval_dataset is not None) else None,
        data_collator=data_collator,
    )

    print(">>> 开始训练...")
    trainer.train()

    # 仅保存 LoRA 适配器
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))

    # 结束 SwanLab 任务
    swanlab.finish()


if __name__ == "__main__":
    main()

    