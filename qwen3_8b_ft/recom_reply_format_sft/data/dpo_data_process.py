import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from gemini_client.gemini_client import GeminiClient

# 质量评估的prompt
QUALITY_COMPARISON_SYSTEM_PROMPT = """你是一个推荐回复质量评估专家，负责比较两组推荐回复的质量。

## 任务
给定对话历史和两组推荐回复（每组包含4个回复），判断哪组回复质量更好。

## 评估标准
1. **相关性**：回复是否与对话上下文相关
2. **自然度**：回复是否自然、真实，不像机器人
3. **吸引力**：回复是否有趣、吸引人，用户更可能点击
4. **多样性**：4个回复是否有不同的风格和角度
5. **语言质量**：语法、拼写、表达是否准确

## 输出格式
只输出 `CHOSEN` 或 `REJECTED`，表示哪组回复更好。
- 如果第一组（chosen）更好，输出 `CHOSEN`
- 如果第二组（rejected）更好，输出 `REJECTED`
- 如果质量相当，输出 `CHOSEN`（优先选择chosen）

## 注意
- 必须严格输出 `CHOSEN` 或 `REJECTED`，不要添加其他文字
- 如果两组质量差异不明显，优先选择第一组（chosen）
"""

QUALITY_COMPARISON_USER_PROMPT = """## 对话历史

{chat_history}

## 第一组推荐回复（chosen）

{chosen_replies}

## 第二组推荐回复（rejected）

{rejected_replies}

## 判断结果（CHOSEN 或 REJECTED）

"""


class DPODataProcessor:
    def __init__(self, max_workers: int = 5, enable_quality_check: bool = True):
        """
        初始化 DPO 数据处理器

        Args:
            max_workers: 最大并发线程数
            enable_quality_check: 是否启用质量检查，确保chosen比rejected更好
        """
        self.max_workers = max_workers
        self.enable_quality_check = enable_quality_check

    def convert_reply_list_to_json(self, reply_list: List[str]) -> Dict[str, str]:
        """
        将原始回复列表转换为JSON格式（与llm_output格式一致）

        Args:
            reply_list: 原始回复列表

        Returns:
            JSON格式的回复字典，包含reply1-4
        """
        # 取前4个回复，如果不足4个则重复最后一个或使用空字符串
        replies = reply_list[:4]
        while len(replies) < 4:
            if replies:
                replies.append(replies[-1])  # 重复最后一个
            else:
                replies.append("")  # 如果没有回复，使用空字符串

        return {
            "reply1": replies[0],
            "reply2": replies[1],
            "reply3": replies[2],
            "reply4": replies[3],
        }

    def extract_chat_history(self, messages: List[Dict]) -> List[Dict]:
        """
        从messages中提取对话历史（移除assistant的回复）

        Args:
            messages: 完整的messages列表（包含assistant回复）

        Returns:
            只包含对话历史的messages（不包含assistant回复）
        """
        # 移除最后一个assistant回复，只保留对话历史
        chat_history = []
        for msg in messages:
            if msg["role"] == "assistant":
                break  # 遇到assistant回复就停止
            chat_history.append(msg)
        return chat_history

    def format_replies_for_comparison(self, replies_json: Dict[str, str]) -> str:
        """
        将回复JSON格式化为可读的字符串，用于质量比较

        Args:
            replies_json: JSON格式的回复字典

        Returns:
            格式化的字符串
        """
        lines = []
        for key in ["reply1", "reply2", "reply3", "reply4"]:
            if key in replies_json:
                lines.append(f"- {key}: {replies_json[key]}")
        return "\n".join(lines)

    def compare_reply_quality(
        self,
        chat_history: List[Dict],
        chosen_replies: Dict[str, str],
        rejected_replies: Dict[str, str],
        judge_client: GeminiClient,
    ) -> bool:
        """
        比较两组回复的质量，判断chosen是否确实比rejected更好

        Args:
            chat_history: 对话历史
            chosen_replies: chosen回复（JSON格式）
            rejected_replies: rejected回复（JSON格式）
            judge_client: GeminiClient实例

        Returns:
            True表示chosen确实更好，False表示rejected更好或质量相当
        """
        # 格式化对话历史
        history_lines = []
        for msg in chat_history:
            if msg["role"] == "system":
                continue  # 跳过system消息，只保留对话内容
            role = msg["role"]
            content = msg["content"]
            history_lines.append(f"{role}: {content}")
        chat_history_str = "\n".join(history_lines)

        # 格式化回复
        chosen_str = self.format_replies_for_comparison(chosen_replies)
        rejected_str = self.format_replies_for_comparison(rejected_replies)

        messages = [
            {"role": "system", "content": QUALITY_COMPARISON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUALITY_COMPARISON_USER_PROMPT.format(
                    chat_history=chat_history_str,
                    chosen_replies=chosen_str,
                    rejected_replies=rejected_str,
                ),
            },
        ]

        max_trials = 3
        for trial in range(max_trials):
            try:
                response, tokens, finish_reason = judge_client.generate(messages=messages)
                response_upper = response.upper().strip()

                if "CHOSEN" in response_upper:
                    return True
                elif "REJECTED" in response_upper:
                    return False
                else:
                    if trial < max_trials - 1:
                        print(
                            f"⚠️ Invalid quality comparison response (trial {trial+1}/{max_trials}): {response[:100]}"
                        )
                    continue
            except Exception as e:
                if trial < max_trials - 1:
                    print(f"⚠️ Error in compare_reply_quality (trial {trial+1}/{max_trials}): {e}")
                continue

        # 如果所有尝试都失败，默认返回True（保守策略，假设chosen更好）
        print(f"⚠️ Failed to compare quality after {max_trials} trials, default to CHOSEN")
        return True

    def process_single_item(
        self,
        idx: int,
        item: Dict,
        output_file: Path,
        stats: Dict,
        stats_lock: Lock,
        file_lock: Lock,
    ) -> None:
        """
        处理单个数据项，转换为DPO格式

        Args:
            idx: 数据项索引
            item: 数据项（从train.jsonl读取）
            output_file: 输出文件路径
            stats: 统计信息字典
            stats_lock: 统计信息锁
            file_lock: 文件写入锁
        """
        # 每个线程创建自己的judge client实例
        judge_client = None
        if self.enable_quality_check:
            judge_client = GeminiClient(config={"model": "gemini-2.5-flash", "temperature": 0.1})

        try:
            # 提取数据
            messages = item["messages"]
            llm_output = item.get("llm_output", {})
            reply_list = item.get("reply_list", [])

            # 检查数据完整性
            if not llm_output or not isinstance(llm_output, dict):
                with stats_lock:
                    stats["invalid_llm_output"] += 1
                return

            if not reply_list or not isinstance(reply_list, list):
                with stats_lock:
                    stats["invalid_reply_list"] += 1
                return

            # 提取对话历史（移除assistant回复）
            chat_history = self.extract_chat_history(messages)

            # 转换回复格式
            chosen_replies = llm_output  # 已经是JSON格式
            rejected_replies = self.convert_reply_list_to_json(reply_list)

            # 质量检查：确保chosen确实比rejected更好
            if self.enable_quality_check and judge_client:
                is_chosen_better = self.compare_reply_quality(
                    chat_history, chosen_replies, rejected_replies, judge_client
                )

                if not is_chosen_better:
                    # rejected更好，交换chosen和rejected
                    chosen_replies, rejected_replies = rejected_replies, chosen_replies
                    with stats_lock:
                        stats["swapped_chosen_rejected"] += 1
                else:
                    with stats_lock:
                        stats["chosen_better"] += 1
            else:
                # 不进行质量检查，直接假设chosen更好
                with stats_lock:
                    stats["no_quality_check"] += 1

            # 构建DPO格式数据
            chosen_json_str = json.dumps(chosen_replies, ensure_ascii=False)
            rejected_json_str = json.dumps(rejected_replies, ensure_ascii=False)

            dpo_item = {
                "messages": chat_history,
                "chosen": {
                    "role": "assistant",
                    "content": chosen_json_str,
                },
                "rejected": {
                    "role": "assistant",
                    "content": rejected_json_str,
                },
            }

            # 线程安全的文件写入
            with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
                    f.flush()

            with stats_lock:
                stats["processed_success"] += 1

        except Exception as e:
            print(f"⚠️ [{idx+1}] Error processing item: {e}")
            with stats_lock:
                stats["processed_failed"] += 1

    def process_dpo_data(self, input_file: str, output_file: str):
        """
        处理DPO数据转换

        Args:
            input_file: 输入的train.jsonl文件路径
            output_file: 输出的DPO格式jsonl文件路径
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果输出文件已存在，清空它
        if output_path.exists():
            output_path.unlink()

        # 读取输入数据
        print(f"[Loading] Reading data from: {input_file}")
        data_items = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data_items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse JSON line: {e}")
                        continue

        total_items = len(data_items)
        print(f"[Loaded] Total items: {total_items}")

        # 线程安全的统计信息和文件写入锁
        stats = {
            "processed_success": 0,
            "processed_failed": 0,
            "invalid_llm_output": 0,
            "invalid_reply_list": 0,
            "chosen_better": 0,
            "swapped_chosen_rejected": 0,
            "no_quality_check": 0,
        }
        stats_lock = Lock()
        file_lock = Lock()

        print(f"[Start] Processing {total_items} items with {self.max_workers} workers")
        print(f"[Output] Saving to: {output_path}")
        if self.enable_quality_check:
            print(f"[Quality Check] Enabled - ensuring chosen is better than rejected")
        else:
            print(f"[Quality Check] Disabled - using chosen/rejected as-is")

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    self.process_single_item, idx, item, output_path, stats, stats_lock, file_lock
                ): idx
                for idx, item in enumerate(data_items)
            }

            # 处理完成的任务并显示进度
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                except Exception as e:
                    print(f"⚠️ [{idx+1}] Task failed with exception: {e}")

                # 每处理 50 条或完成时打印进度
                if completed % 50 == 0 or completed == total_items:
                    with stats_lock:
                        current_stats = stats.copy()
                    print(f"[Progress] {completed}/{total_items} | {current_stats}")

        # 最终统计信息
        print(f"\n[Final Stats] {stats}")
        print(f"[Output] Saved to: {output_path}")
        print(f"[Success Rate] {stats['processed_success']}/{total_items} ({stats['processed_success']/total_items*100:.2f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SFT training data to DPO format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input train.jsonl file path (output from reply_rewrite.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output DPO format jsonl file path",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent threads (default: 5)",
    )
    parser.add_argument(
        "--disable-quality-check",
        action="store_true",
        help="Disable quality check (faster but less reliable)",
    )
    args = parser.parse_args()

    processor = DPODataProcessor(
        max_workers=args.max_workers, enable_quality_check=not args.disable_quality_check
    )
    processor.process_dpo_data(args.input, args.output)

