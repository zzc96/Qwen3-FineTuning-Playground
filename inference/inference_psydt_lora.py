# LLM_fine_turning/inference_psydt_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import json
from typing import List, Dict

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业、共情的心理咨询助理。你的回复应："
    "1) 共情与理解；2) 关注来访者的情绪与需要；"
    "3) 避免诊断与评判；4) 提供支持性建议与可执行的小步骤；"
    "5) 鼓励在需要时寻求专业帮助。"
)

class PsyDTChatbot:
    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_turns: int = 6
    ):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.system_prompt = system_prompt
        self.history_turns = max(0, history_turns)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.history: List[Dict[str, str]] = []
        self.reset()

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def load_model(self, merge_lora: bool = False):
        print("🚀 正在加载模型 ...")
        print(f"--> 基座模型: {self.base_model_path}")
        print(f"--> LoRA 适配器: {self.lora_adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, model_id=self.lora_adapter_path)

        if merge_lora:
            self.model = self.model.merge_and_unload()

        self.model.eval()
        print(f"✅ 模型加载完成，设备: {self.model.device}")

    def _trim_history(self):
        # 仅保留最近 N 轮 user/assistant（不含 system）
        if self.history_turns <= 0:
            return
        sys = self.history[0] if self.history and self.history[0]["role"] == "system" else None
        msgs = self.history[1:] if sys else self.history[:]
        # 每轮包含 user+assistant 两条；按末尾截取
        kept = []
        ua = 0
        for m in reversed(msgs):
            kept.append(m)
            if m["role"] == "assistant":
                ua += 1
                if ua >= self.history_turns:
                    break
        kept = list(reversed(kept))
        self.history = ([sys] if sys else []) + kept

    @torch.no_grad()
    def chat(
        self,
        user_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        prompt_text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        self.history.append({"role": "assistant", "content": response})
        return response


def run_interactive(bot: PsyDTChatbot, args):
    print("\n" + "="*80)
    print("🎯 心理咨询多轮对话 - 交互模式\n输入 'exit' 或 'quit' 退出，会话使用多轮记忆。")
    print("="*80)

    while True:
        try:
            user_input = input("\n👤 来访者: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue
            print("🤖 咨询助理: ", end="", flush=True)
            reply = bot.chat(
                user_input,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            print(reply)
        except (KeyboardInterrupt, EOFError):
            break
    print("\n👋 已结束对话")

def run_test(bot: PsyDTChatbot, args):
    test_questions = [
        "最近压力很大，总是睡不好，怎么办？",
        "我经常对自己很苛刻，觉得自己不够好。",
        "和家人的沟通总是会吵起来，我不想这样。",
        "我有点焦虑，害怕在公众场合表达。",
        "能给我一些缓解焦虑的小练习吗？"
    ]
    results = []
    print("\n" + "="*80)
    print("🧪 批量测试开始")
    print("="*80)
    for i, q in enumerate(test_questions, 1):
        print(f"\n📝 {i}/{len(test_questions)} 题目: {q}\n" + "-"*60)
        resp = bot.chat(
            q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(f"🤖 回复: {resp}")
        results.append({"question": q, "response": resp})

    if args.test_output_file:
        with open(args.test_output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 测试完成，结果已保存至: {args.test_output_file}")

def main():
    parser = argparse.ArgumentParser(description="PsyDT LoRA 多轮对话推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径，如 ./Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA 适配器路径，如 ./output/qwen3-1_7b-psydt-lora/lora_adapter")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"])
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--history_turns", type=int, default=6, help="保留最近 N 轮对话记忆")
    parser.add_argument("--merge_lora", action="store_true", help="将 LoRA 合并进基座模型以加速推理")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--test_output_file", type=str, default="psydt_sft_test_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"基础模型路径不存在: {args.model_path}")
    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"LoRA 适配器路径不存在: {args.adapter_path}")

    bot = PsyDTChatbot(
        base_model_path=args.model_path,
        lora_adapter_path=args.adapter_path,
        system_prompt=args.system_prompt,
        history_turns=args.history_turns
    )
    bot.load_model(merge_lora=args.merge_lora)

    if args.mode == "interactive":
        run_interactive(bot, args)
    else:
        run_test(bot, args)

if __name__ == "__main__":
    main()