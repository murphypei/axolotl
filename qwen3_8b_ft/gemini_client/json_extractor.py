#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 提取器
从 LLM 输出中智能提取 JSON 内容，兼容各种可能的输出格式
"""

import json
import re
from typing import Any, Dict, List, Optional, Union


class JSONExtractor:
    """
    JSON 提取器

    支持从以下格式中提取 JSON：
    1. Markdown 代码块: ```json\n{...}\n```
    2. Markdown 代码块(无语言标记): ```\n{...}\n```
    3. 纯 JSON: {...}
    4. 带前后文本的 JSON: "Here is the result: {...} Hope this helps!"
    5. 转义的 JSON: \\{...\\}
    6. 多个 JSON 对象（返回第一个或全部）
    """

    def __init__(self):
        """初始化 JSON 提取器"""
        # Markdown 代码块模式（支持 json、JSON、javascript 等语言标记）
        self.markdown_patterns = [
            # ```json ... ```
            r"```\s*(?:json|JSON)\s*\n(.*?)\n```",
            # ```javascript ... ```
            r"```\s*(?:javascript|js|JS)\s*\n(.*?)\n```",
            # ``` ... ```
            r"```\s*\n?(.*?)\n?```",
        ]

        # JSON 对象模式（匹配 {...}）
        self.json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

        # JSON 数组模式（匹配 [...]）
        self.json_array_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"

    def extract(self, text: str, extract_first: bool = True, strict: bool = False) -> Optional[Union[Dict, List, str]]:
        """
        从文本中提取 JSON

        Args:
            text: 包含 JSON 的文本
            extract_first: 如果找到多个 JSON，是否只返回第一个（默认 True）
            strict: 严格模式，要求 JSON 必须是有效的（默认 False）

        Returns:
            解析后的 JSON 对象（dict/list），如果失败返回 None
        """
        if not text or not isinstance(text, str):
            return None

        # 尝试方法1: 提取 Markdown 代码块中的 JSON
        json_text = self._extract_from_markdown(text)
        if json_text:
            parsed = self._parse_json(json_text, strict)
            if parsed is not None:
                return parsed

        # 尝试方法2: 直接解析整个文本（可能是纯 JSON）
        parsed = self._parse_json(text.strip(), strict)
        if parsed is not None:
            return parsed

        # 尝试方法3: 查找文本中的 JSON 对象或数组
        json_candidates = self._find_json_in_text(text)
        if json_candidates:
            if extract_first:
                # 返回第一个成功解析的 JSON
                for candidate in json_candidates:
                    parsed = self._parse_json(candidate, strict)
                    if parsed is not None:
                        return parsed
            else:
                # 返回所有成功解析的 JSON
                results = []
                for candidate in json_candidates:
                    parsed = self._parse_json(candidate, strict)
                    if parsed is not None:
                        results.append(parsed)
                return results if results else None

        return None

    def extract_as_string(self, text: str) -> Optional[str]:
        """
        从文本中提取 JSON 字符串（不解析）

        Args:
            text: 包含 JSON 的文本

        Returns:
            JSON 字符串，如果失败返回 None
        """
        if not text or not isinstance(text, str):
            return None

        # 尝试从 Markdown 代码块提取
        json_text = self._extract_from_markdown(text)
        if json_text and self._is_valid_json(json_text):
            return json_text

        # 尝试查找 JSON 对象或数组
        json_candidates = self._find_json_in_text(text)
        if json_candidates:
            for candidate in json_candidates:
                if self._is_valid_json(candidate):
                    return candidate

        return None

    def _extract_from_markdown(self, text: str) -> Optional[str]:
        """
        从 Markdown 代码块中提取内容

        Args:
            text: 可能包含 Markdown 代码块的文本

        Returns:
            提取的内容，如果没有找到返回 None
        """
        for pattern in self.markdown_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    return content
        return None

    def _find_json_in_text(self, text: str) -> List[str]:
        """
        在文本中查找所有可能的 JSON 对象和数组

        Args:
            text: 文本内容

        Returns:
            JSON 候选字符串列表
        """
        candidates = []

        # 查找所有 JSON 对象 {...}
        # 使用递归模式匹配嵌套的花括号
        brace_depth = 0
        start_pos = -1

        for i, char in enumerate(text):
            if char == "{":
                if brace_depth == 0:
                    start_pos = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and start_pos != -1:
                    candidates.append(text[start_pos : i + 1])
                    start_pos = -1

        # 查找所有 JSON 数组 [...]
        bracket_depth = 0
        start_pos = -1

        for i, char in enumerate(text):
            if char == "[":
                if bracket_depth == 0:
                    start_pos = i
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth == 0 and start_pos != -1:
                    candidates.append(text[start_pos : i + 1])
                    start_pos = -1

        return candidates

    def _parse_json(self, json_text: str, strict: bool = False) -> Optional[Union[Dict, List]]:
        """
        解析 JSON 字符串

        Args:
            json_text: JSON 字符串
            strict: 严格模式

        Returns:
            解析后的 JSON 对象，失败返回 None
        """
        if not json_text:
            return None

        try:
            # 尝试直接解析
            parsed = json.loads(json_text)
            return parsed
        except json.JSONDecodeError:
            if strict:
                return None

            # 非严格模式：尝试修复常见问题
            # 1. 移除前后空白
            json_text = json_text.strip()

            # 2. 处理转义的引号
            json_text = json_text.replace('\\"', '"')

            # 3. 处理单引号（非标准但常见）
            json_text = json_text.replace("'", '"')

            # 4. 移除尾部逗号（非标准但常见）
            json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)

            # 5. 处理 Python 风格的布尔值和 None
            json_text = re.sub(r"\bTrue\b", "true", json_text)
            json_text = re.sub(r"\bFalse\b", "false", json_text)
            json_text = re.sub(r"\bNone\b", "null", json_text)

            try:
                parsed = json.loads(json_text)
                return parsed
            except json.JSONDecodeError:
                return None

    def _is_valid_json(self, json_text: str) -> bool:
        """
        检查字符串是否为有效的 JSON

        Args:
            json_text: JSON 字符串

        Returns:
            True/False
        """
        try:
            json.loads(json_text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def extract_with_fallback(self, text: str, default: Any = None) -> Union[Dict, List, Any]:
        """
        提取 JSON，失败时返回默认值

        Args:
            text: 包含 JSON 的文本
            default: 提取失败时的默认值

        Returns:
            解析后的 JSON 或默认值
        """
        result = self.extract(text)
        return result if result is not None else default


# 便捷函数
def extract_json(text: str, extract_first: bool = True, strict: bool = False) -> Optional[Union[Dict, List]]:
    """
    从文本中提取 JSON（便捷函数）

    Args:
        text: 包含 JSON 的文本
        extract_first: 如果找到多个 JSON，是否只返回第一个
        strict: 严格模式

    Returns:
        解析后的 JSON 对象
    """
    extractor = JSONExtractor()
    return extractor.extract(text, extract_first=extract_first, strict=strict)


def extract_json_string(text: str) -> Optional[str]:
    """
    从文本中提取 JSON 字符串（便捷函数）

    Args:
        text: 包含 JSON 的文本

    Returns:
        JSON 字符串
    """
    extractor = JSONExtractor()
    return extractor.extract_as_string(text)


if __name__ == "__main__":
    """测试 JSON 提取器"""

    test_cases = [
        {
            "name": "纯 JSON",
            "text": '{"name": "Alice", "age": 30}',
        },
        {
            "name": "Markdown 代码块（带 json 标记）",
            "text": """```json
{
    "name": "Bob",
    "age": 25,
    "hobbies": ["reading", "coding"]
}
```""",
        },
        {
            "name": "Markdown 代码块（无语言标记）",
            "text": """```
{"status": "success", "data": {"count": 10}}
```""",
        },
        {
            "name": "带前后文本的 JSON",
            "text": 'Here is the result: {"result": "ok", "value": 42} Hope this helps!',
        },
        {
            "name": "单引号 JSON（非标准）",
            "text": "{'name': 'Charlie', 'score': 95}",
        },
        {
            "name": "带尾部逗号的 JSON（非标准）",
            "text": '{"x": 1, "y": 2,}',
        },
        {
            "name": "Python 风格的布尔值",
            "text": '{"active": True, "deleted": False, "data": None}',
        },
        {
            "name": "嵌套 JSON",
            "text": """The API response is:
```json
{
    "user": {
        "id": 123,
        "profile": {
            "name": "David",
            "email": "david@example.com"
        }
    },
    "status": "active"
}
```
This is the complete data.""",
        },
        {
            "name": "多个 JSON 对象",
            "text": '{"first": 1} and also {"second": 2}',
        },
        {
            "name": "JSON 数组",
            "text": "[1, 2, 3, 4, 5]",
        },
        {
            "name": "无效格式",
            "text": "This is not JSON at all, just plain text.",
        },
    ]

    print("=" * 80)
    print("🧪 Testing JSON Extractor")
    print("=" * 80 + "\n")

    extractor = JSONExtractor()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 80)
        print(f"Input: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
        print()

        # 测试提取
        result = extractor.extract(test_case["text"], strict=False)

        if result is not None:
            print(f"✅ 提取成功:")
            print(f"   类型: {type(result).__name__}")
            print(f"   内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"❌ 提取失败")

        # 测试提取字符串
        json_str = extractor.extract_as_string(test_case["text"])
        if json_str:
            print(f"   字符串形式: {json_str[:100]}{'...' if len(json_str) > 100 else ''}")

    print(f"\n{'=' * 80}")
    print(f"✅ All tests completed")
    print(f"{'=' * 80}\n")
