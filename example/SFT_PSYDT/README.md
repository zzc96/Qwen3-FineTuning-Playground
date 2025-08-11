## 用开源心理学数据微调大语言模型（LoRA + Qwen3）

本教程演示如何使用开源心理学多轮对话数据（PsyDTCorpus）对大模型进行 LoRA 微调，并进行推理与测试。脚本支持自动下载模型与数据，默认仅使用训练集 300 条、测试集 20 条用于快速验证。

---

### 依赖环境

- Python 3.10+
- PyTorch（建议带 CUDA）
- transformers, datasets, peft, modelscope, swanlab
- 建议 GPU 显存 ≥ 16GB（Qwen3-1.7B + LoRA 一般 12–16GB 可运行）

安装示例：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 按需选择CUDA
pip install transformers datasets peft modelscope swanlab accelerate
```

---

### 数据集简介

- 使用 `YIRONGCHEN/PsyDTCorpus`（多轮心理咨询对话，JSON 数组）
- 每条数据形如：
```json
[
  {
    "messages": [
      {"role": "system", "content": "系统提示（可选）"},
      {"role": "user", "content": "来访者的话"},
      {"role": "assistant", "content": "咨询助理回复"},
      ...
    ]
  }
]
```
- 训练脚本会将多轮对话拆分为多个样本，仅对 `assistant` 角色位置计算 loss。

---

### 快速开始（自动下载 + 小样本训练）

以下命令将：
- 自动检查并下载基础模型 `Qwen/Qwen3-1.7B`
- 自动下载所需的 PsyDT 数据文件（若本地不存在）
- 仅使用训练集前 300 条、测试集前 20 条对话进行训练与评估
- 输出 LoRA 适配器到 `./output/qwen3-1_7b-psydt-lora/lora_adapter`

```bash
python LLM_fine_turning/train_psydt_lora.py
```

可选参数（覆盖默认值）：
```bash
python LLM_fine_turning/train_psydt_lora.py \
  --max_train_items 300 \
  --max_eval_items 20 \
  --output_dir ./output/qwen3-1_7b-psydt-lora \
  --model_local_dir ./Qwen/Qwen3-1.7B \
  --dataset_repo YIRONGCHEN/PsyDTCorpus \
  --dataset_dir ./data/PsyDTCorpus
```

说明：
- 脚本默认开启评估（`--do_eval` 默认 True）。
- 首次使用 modelscope 可能需要配置镜像或登录，网络慢时可尝试代理。
- 训练过程会将日志上报到 SwanLab（`SWANLAB_PROJECT` 环境变量已在脚本内配置）。

---

### 推理与测试（多轮心理咨询对话）

推理脚本：`LLM_fine_turning/inference_psydt_lora.py`  
功能：多轮记忆、交互式与批量测试两种模式、可选合并 LoRA（加速推理）

交互式对话：
```bash
python LLM_fine_turning/inference_psydt_lora.py \
  --model_path "./Qwen/Qwen3-1.7B" \
  --adapter_path "./output/qwen3-1_7b-psydt-lora/lora_adapter" \
  --mode interactive
```

批量测试并保存结果：
```bash
python LLM_fine_turning/inference_psydt_lora.py \
  --model_path "./Qwen/Qwen3-1.7B" \
  --adapter_path "./output/qwen3-1_7b-psydt-lora/lora_adapter" \
  --mode test \
  --test_output_file "psydt_results.json"
```

常用可选项：
- `--history_turns 6`：保留最近 N 轮对话记忆
- `--merge_lora`：合并 LoRA 到基础模型，推理更快（显存更高）
- `--system_prompt`：自定义心理咨询风格提示词

---

### 训练脚本要点

- 文件：`LLM_fine_turning/train_psydt_lora.py`
- 模型：默认 `Qwen/Qwen3-1.7B`，自动下载到 `./Qwen/Qwen3-1.7B`
- LoRA：`r=8, alpha=32, dropout=0.1`，目标模块包含 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- 数据：若 `--train_file` / `--eval_file` 缺失，自动从 `YIRONGCHEN/PsyDTCorpus` 下载对应 JSON
- 仅对 `assistant` 的 token 计算 loss，使用聊天模板构建上下文前缀

重要参数：
- `--max_length`：序列最大长度（默认 2048）
- `--per_device_train_batch_size`、`--gradient_accumulation_steps`：按显存调整
- `--num_train_epochs`、`--learning_rate`、`--save_steps`、`--eval_steps`
- `--max_train_items`、`--max_eval_items`：本教程默认 300/20

产物：
- LoRA 适配器保存在 `./output/qwen3-1_7b-psydt-lora/lora_adapter`

---

### 常见问题

- 显存不足
  - 降低 `--per_device_train_batch_size`，提高 `--gradient_accumulation_steps`
  - 开启 `gradient_checkpointing`（脚本默认）
  - 使用更小模型或更短 `--max_length`
- bfloat16 不支持
  - 无 bfloat16 的 GPU 会自动回退为 float32；可手动指定或升级硬件
- modelscope 下载慢/失败
  - 配置镜像/代理，或提前手动下载到 `--dataset_dir` / `--model_local_dir`
- pad_token 报错
  - 推理脚本已在需要时将 `pad_token = eos_token`

---

### 伦理与合规声明

- 心理学对话涉及敏感内容。本项目仅用于教学与研究，不替代专业心理咨询。
- 请遵守数据集与模型许可证，并在应用中加入必要的安全防护与免责声明。

---

### 引用与致谢

- 模型：Qwen/Qwen3-1.7B
- 数据：YIRONGCHEN/PsyDTCorpus
- 训练与推理：基于 Hugging Face Transformers + PEFT LoRA + SwanLab