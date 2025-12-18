import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from threading import Lock
from typing import Dict, List

current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from gemini_client.gemini_client import GeminiClient
from json_repair import repair_json

FILTER_VALID_CHAT_SYSTEM_PROMPT = """你是一个IM对话研究专家，负责识别一段 chat history 是否是有效的互动对话。输出 `YES` 或 `NO` 作为判断结果。
## 对话示例

### 无效对话示例 1

```
User1: أنا أشارك في حفلة دردشة عبر Saya. 🥳 مستمتع جدًا بالدردشة ولعب ألعاب حفلة كثيرة في نفس الوقت!🎶 انضم لي الآن！
User1: أنا أشارك في حفلة دردشة عبر Saya. 🥳 مستمتع جدًا بالدردشة ولعب ألعاب حفلة كثيرة في نفس الوقت!🎶 انضم لي الآن！
user2: أنا أشارك في حفلة دردشة عبر Saya. 🥳 مستمتع جدًا بالدردشة ولعب ألعاب حفلة كثيرة في نفس الوقت!🎶 انضم لي الآن！
user2: أنا أشارك في حفلة دردشة عبر Saya. 🥳 مستمتع جدًا بالدردشة ولعب ألعاب حفلة كثيرة في نفس الوقت!🎶 انضم لي الآن！
````

**原因**：两个人都是机械的复制同一句打招呼，并没有实质性的互动。

### 无效对话示例 2

```
User1: يالله بخرج العمل
User1: باااااااي
user2: الحمدلله عالسلامة 😊
User1: اممممممممح 
user2: بايات
User1: اممممممممح 
User1: اممممممممح 
User1: اممممممممح 
User1: اممممممممح 
User1: اممممممممح 
user2: احيه
```

**原因**：User1 一直在重复同一个回复，用户基本没交流。

### 有效对话示例

```
user1: I'm good, just relaxing a bit after streaming.
User2: i ma like song
User1: i miss you 
user2: That's cool! What kind of songs do you like?
User1: yes
user2: Any particular genre you like to listen to?
User1: yes
User1: i love you 💗
user2: Who's the artist or song?
user1: Aw, that's sweet! Come join my voice room.
```
**原因**：User1 和 User2 进行了多次互动，对话内容丰富，有实质性的交流，并且有多个回复。
"""

FILTER_VALID_CHAT_USER_PROMPT = """## chat history

{chat_history}

## Judge result (YES or NO)

"""

SYSTEM_PROMPT = """## Task
You are helping a user on SAYA app generate recommended replies based on their chat history.

## Goal
Generate 4 replies (5-15 words each) that:
1. Fit the conversation context
2. Feel authentic and casual
3. Help continue the dialogue naturally
4. Each reply no more than 20 words.

**IMPORTANT**: 
- The first 2 replies (reply1 and reply2) are shown directly to users. Focus on making these two the most appealing and likely to be clicked.
- If the last message in the chat history is from yourself, then you shold generate greetings replies for the user.

## Reply Priority
- **reply1 & reply2**: Must be your BEST, most click-worthy responses. These are the first impression.
- **reply3 & reply4**: Good alternatives with different styles, but secondary priority.

## Output Format
Return exactly 4 replies in this JSON format:
```json
{
    "reply1": "Your BEST reply - high click potential",
    "reply2": "Your 2nd BEST reply - different style from reply1",
    "reply3": "Alternative reply - different angle",
    "reply4": "Alternative reply - backup option"
}
```

## Rules
- **CRITICAL**: All responses MUST be written entirely in the user's language.
- NEVER REVEAL YOURSELF AS AN AI/AI-ASSISTANT/ROBOT.
- Make reply1 and reply2 especially appealing (they're shown first!)
- Ensure reply1 and reply2 have different styles/tones
- Keep all replies natural and conversational, avoid robotic templates
- Avoid explicit, sexual, or religious topics

"""

USER_PROMPT = """## chat history

{chat_history}

## Generated the {role} replies or greetings

"""


class ReplyRewrite:
    def __init__(self, max_workers: int = 5):
        """
        初始化 ReplyRewrite

        Args:
            max_workers: 最大并发线程数，默认 5（可根据 API 速率限制调整）
        """
        self.max_workers = max_workers
        # 注意：每个线程会创建自己的 client 实例，避免共享状态

    def load_data(self, raw_data_path: str):
        valid_chat_list = []
        stats = {
            "unknown_user_uid": 0,
            "valid_chat": 0,
        }

        with open(raw_data_path, "r") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            # print(f"Processing: {idx+1} / {len(data)}")
            user_uid = item["user_uid"]
            robot_uid = item["robot_uid"]
            chat_history = item["chat_history"]

            valid = True
            messages = []
            for chat_turn in chat_history:
                if chat_turn["uid"] == user_uid:
                    messages.append({"role": "user1", "content": chat_turn["content"]})
                elif chat_turn["uid"] == robot_uid:
                    messages.append({"role": "user2", "content": chat_turn["content"]})
                else:
                    # print(f"⚠️ Unknown user uid: {chat_turn['uid']}")
                    valid = False

            if valid:
                # NOTE: robot 就是 assistant 的角色，此处是 user2
                valid_chat_list.append(
                    {"messages": messages, "reply_list": item["reply_list"], "assistant_role": "user2"}
                )
                stats["valid_chat"] += 1
            else:
                stats["unknown_user_uid"] += 1

        print(stats)
        return valid_chat_list

    def build_llm_messages(self, chat_history: List[Dict], assistant_role: str) -> List[Dict]:
        user_messages = []
        for chat_turn in chat_history:
            user_messages.append(f"{chat_turn['role']}: {chat_turn['content']}")
        user_messages = "\n".join(user_messages)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(chat_history=user_messages, role=assistant_role)},
        ]
        return messages

    def judge_valid_chat(self, chat_history: List[Dict], judge_client: GeminiClient) -> bool:
        """
        判断对话是否有效

        Args:
            chat_history: 对话历史
            judge_client: GeminiClient 实例（每个线程使用独立的实例）

        Returns:
            bool: 对话是否有效
        """
        user_messages = []
        for chat_turn in chat_history:
            user_messages.append(f"{chat_turn['role']}: {chat_turn['content']}")
        user_messages = "\n".join(user_messages)

        messages = [
            {"role": "system", "content": FILTER_VALID_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": FILTER_VALID_CHAT_USER_PROMPT.format(chat_history=user_messages)},
        ]

        max_trials = 3
        for trial in range(max_trials):
            try:
                response, tokens, finish_reason = judge_client.generate(messages=messages)
                response_lower = response.lower().strip()
                if "yes" in response_lower:
                    return True
                elif "no" in response_lower:
                    return False
                else:
                    if trial < max_trials - 1:
                        print(f"⚠️ Invalid judge response (trial {trial+1}/{max_trials}): {response[:100]}")
                    continue
            except Exception as e:
                if trial < max_trials - 1:
                    print(f"⚠️ Error in judge_valid_chat (trial {trial+1}/{max_trials}): {e}")
                continue

        # 如果所有尝试都失败，默认返回 False（保守策略）
        print(f"⚠️ Failed to judge valid chat after {max_trials} trials, default to False")
        return False

    def process_single_item(
        self, idx: int, item: Dict, output_file: Path, stats: Dict, stats_lock: Lock, file_lock: Lock
    ) -> None:
        """
        处理单个数据项（线程安全）

        Args:
            idx: 数据项索引
            item: 数据项
            output_file: 输出文件路径
            stats: 统计信息字典
            stats_lock: 统计信息锁
            file_lock: 文件写入锁
        """
        # 每个线程创建自己的 client 实例，避免共享状态和线程安全问题
        judge_client = GeminiClient(config={"model": "gemini-2.5-flash", "temperature": 0.1})
        reply_client = GeminiClient(config={"model": "gemini-2.5-pro", "temperature": 0.3})

        try:
            with stats_lock:
                stats["processed_num"] += 1

            # 步骤1: 判断对话是否有效
            judge_result = self.judge_valid_chat(item["messages"], judge_client)
            if not judge_result:
                return

            with stats_lock:
                stats["judge_yes"] += 1

            # 步骤2: 生成推荐回复
            processed_result = None
            max_trials = 3
            for trial in range(max_trials):
                try:
                    gemini_messages = self.build_llm_messages(item["messages"], item["assistant_role"])
                    response, tokens, finish_reason = reply_client.generate(
                        messages=gemini_messages, response_mime_type="application/json"
                    )

                    # json_result = self.json_extractor.extract(response)

                    json_result = repair_json(response)
                    if not json_result:
                        if trial < max_trials - 1:
                            print(f"⚠️ [{idx+1}] Failed to parse JSON (trial {trial+1}/{max_trials}): {response[:200]}")
                        continue

                    # 将 JSON 对象转换为字符串，用于 SFT 训练
                    json_content = json.dumps(json_result, ensure_ascii=False)

                    processed_result = {
                        "messages": gemini_messages + [{"role": "assistant", "content": json_content}],
                        "reply_list": item["reply_list"],
                        "llm_output": json_result,
                        "reply_model": "gemini-2.5-pro",
                        "judge_model": "gemini-2.5-flash",
                    }

                    break
                except Exception as e:
                    if trial < max_trials - 1:
                        print(f"⚠️ [{idx+1}] Error generating reply (trial {trial+1}/{max_trials}): {e}")
                    continue

            if processed_result:
                # 线程安全的文件写入
                with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(processed_result, ensure_ascii=False) + "\n")
                        f.flush()

                with stats_lock:
                    stats["processed_success"] += 1
            else:
                print(f"⚠️ [{idx+1}] Failed to process chat after {max_trials} trials")
                with stats_lock:
                    stats["processed_failed"] += 1

        except Exception as e:
            print(f"⚠️ [{idx+1}] Unexpected error: {e}")
            with stats_lock:
                stats["processed_failed"] += 1

    def rewrite_train_data(self):
        """
        多线程并行处理数据
        """
        valid_chat_list = self.load_data("/mnt/cephfs2/peichao/code/Lumitune/data/turn_ge6_2025-11-25/valid_data.json")
        total_items = len(valid_chat_list)

        # 线程安全的统计信息和文件写入锁
        stats = {
            "judge_yes": 0,
            "processed_num": 0,
            "processed_failed": 0,
            "processed_success": 0,
        }
        stats_lock = Lock()
        file_lock = Lock()

        # 确保输出目录存在
        output_dir = current_dir / "recom_reply_turn_ge6_2025-11-25"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "train.jsonl"

        print(f"[Start] Processing {total_items} items with {self.max_workers} workers")
        print(f"[Output] Saving to: {output_file}")

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_single_item, idx, item, output_file, stats, stats_lock, file_lock): idx
                for idx, item in enumerate(valid_chat_list)
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

                # 每处理 10 条或完成时打印进度
                if completed % 10 == 0 or completed == total_items:
                    with stats_lock:
                        current_stats = stats.copy()
                    print(f"[Progress] {completed}/{total_items} | {current_stats}")

        # 最终统计信息
        print(f"\n[Final Stats] {stats}")
        print(f"[Output] Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate recommended replies using Gemini API")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent threads (default: 5). Adjust based on API rate limits.",
    )
    args = parser.parse_args()

    rw = ReplyRewrite(max_workers=args.max_workers)
    rw.rewrite_train_data()
