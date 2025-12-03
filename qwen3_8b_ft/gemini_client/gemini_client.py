#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的 Gemini LLM 客户端
用于模型评估
"""

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

current_dir = Path(__file__).resolve().parent

# Gemini 默认配置
DEFAULT_GEMINI_CONFIG = {
    # LLM提供商
    "provider": "gemini",
    # Gemini基础配置
    "project_id": "bigolive-ai-chat",  # GCP项目ID
    "location": "us-central1",  # Gemini API 区域
    "model": "gemini-2.5-pro",  # 评估使用的模型
    "credentials_path": f"{current_dir}/gemini_config.json",  # GCP服务账号密钥路径
    # 超时和重试配置
    "timeout": 300,  # 超时时间（秒）
    "max_retries": 3,  # 失败重试次数
    # 生成参数配置
    "temperature": 0.7,  # 生成温度
    "max_tokens": 4096,  # 最大输出token数
    "thinking_budget": 128,  # 思考预算
    # 安全设置
    "safety_settings": [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ],
    # 计费标签
    "billing_name": "peichao.murphy|saya-aichat-service",
    # 评估配置
    "batch_size": 4,  # 评估时的批处理大小
    "retry_delay": 5,  # 重试延迟（秒）
}


class GeminiClient:
    """Gemini 评估客户端（简化版，用于模型评估）"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 Gemini 客户端

        Args:
            config: Gemini 配置字典（可选）。如果提供，会覆盖默认配置
        """
        # 合并配置：默认配置 + 用户配置
        self.config = DEFAULT_GEMINI_CONFIG.copy()
        if config:
            self.config.update(config)

        self._init_client()

    def _init_client(self):
        """初始化 Gemini 客户端"""
        print("=" * 70)
        print("🤖 Initializing Gemini Client")
        print("=" * 70 + "\n")

        try:
            # 读取配置
            self.project_id = self.config["project_id"]
            self.location = self.config["location"]
            self.model_name = self.config["model"]
            self.temperature = self.config.get("temperature", 0.0)  # 评估时使用低温度
            self.max_tokens = self.config.get("max_tokens", 4096)
            self.timeout = self.config.get("timeout", 60)

            print(f"  Project: {self.project_id}")
            print(f"  Location: {self.location}")
            print(f"  Model: {self.model_name}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Max tokens: {self.max_tokens}")

            # 设置 Google 应用凭据
            credentials_path = self.config.get("credentials_path")
            if credentials_path and os.path.exists(credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                print(f"\n  ✅ Credentials loaded: {credentials_path}")
            else:
                print(f"\n  ⚠️  Credentials not found: {credentials_path}")

            # 初始化客户端
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )

            print(f"\n✅ Gemini client initialized successfully")
            print()

        except Exception as e:
            print(f"\n❌ Failed to initialize Gemini client: {e}")
            traceback.print_exc()
            print()
            raise

    def _convert_messages_to_contents(self, messages: List[Dict]) -> Tuple[List, Optional[List]]:
        """
        转换标准消息格式为 Gemini Content 格式

        Args:
            messages: 标准消息列表 [{"role": "user/system", "content": "..."}]

        Returns:
            (contents, system_instruction) 元组
        """
        contents = []
        system_instruction = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # 系统消息作为 system_instruction
                system_instruction = [types.Part.from_text(text=content)]
            elif role in ["user", "model", "assistant"]:
                # 将 assistant 映射为 model
                if role == "assistant":
                    role = "model"

                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content)]))

        return contents, system_instruction

    def generate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None,
    ) -> Tuple[str, int, str]:
        """
        生成回复

        Args:
            messages: 消息列表
            temperature: 温度参数（None 则使用配置）
            max_tokens: 最大输出 token 数（None 则使用配置）
            response_mime_type: 响应格式（如 "application/json"）

        Returns:
            (response_text, total_tokens, finish_reason) 元组
        """
        if not messages:
            return "", 0, "invalid_request"

        try:
            # 转换消息格式
            contents, system_instruction = self._convert_messages_to_contents(messages)

            # 使用传入参数或默认配置
            request_temp = temperature if temperature is not None else self.temperature
            request_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

            # 配置生成参数
            config_params = {
                "temperature": request_temp,
                "max_output_tokens": request_max_tokens,
            }

            # 如果指定了响应格式
            if response_mime_type:
                config_params["response_mime_type"] = response_mime_type

            generate_content_config = types.GenerateContentConfig(**config_params)

            # 添加系统指令
            if system_instruction:
                generate_content_config.system_instruction = system_instruction

            start_time = time.time()

            # 调用 API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            # print(f"[DEBUG] gemini response: {response}")

            # 处理响应
            end_time = time.time()
            response_time = end_time - start_time

            # 统计 token 使用
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = input_tokens + output_tokens

            if response and response.text:
                return response.text, total_tokens, "stop"
            else:
                return "", 0, "empty_response"

        except google_exceptions.DeadlineExceeded:
            print("gemini error: deadline exceeded")
            return "", 0, "timeout"

        except google_exceptions.PermissionDenied as e:
            print("gemini error: permission denied")
            return "", 0, "permission_denied"

        except google_exceptions.InvalidArgument as e:
            print("gemini error: invalid argument")
            return "", 0, "invalid_argument"

        except google_exceptions.ResourceExhausted as e:
            print("gemini error: resource exhausted")
            return "", 0, "resource_exhausted"

        except Exception as e:
            print(f"gemini error: {e}")
            return "", 0, "error"


if __name__ == "__main__":
    """测试 Gemini 客户端"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试 Gemini 客户端")
    parser.add_argument("--model", type=str, help="模型名称（可选，覆盖默认配置）")
    parser.add_argument("--temperature", type=float, help="温度参数（可选，覆盖默认配置）")
    parser.add_argument("--credentials", type=str, help="凭证文件路径（可选，覆盖默认配置）")
    args = parser.parse_args()

    # 构建自定义配置（覆盖默认值）
    custom_config = {}
    if args.model:
        custom_config["model"] = args.model
    if args.temperature is not None:
        custom_config["temperature"] = args.temperature
    if args.credentials:
        custom_config["credentials_path"] = args.credentials

    print("\n" + "=" * 70)
    print("🧪 Testing Gemini Client")
    print("=" * 70 + "\n")

    print("  Configuration:")
    print(f"    Using default config + custom overrides")
    if custom_config:
        print(f"    Custom config: {custom_config}")
    print()

    # 初始化客户端（使用默认配置 + 自定义覆盖）
    client = GeminiClient(config=custom_config if custom_config else None)

    # 测试调用
    print("=" * 70)
    print("🚀 Testing API Call")
    print("=" * 70 + "\n")

    # messages = [{"role": "user", "content": 'Hello! Please respond with a JSON: {"greeting": "your greeting here"}'}]
    import json
    from pprint import pprint

    with open("qwen3_8b_ft/recom_reply_format_sft/data/recom_reply_turn_ge6_2025-11-25/train.jsonl", "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            pprint(data)
            messages = [data["messages"][0]]
            break

    print("-" * 60)
    pprint(messages)
    response, tokens, finish_reason = client.generate(messages=messages, response_mime_type="application/json")

    print("\n" + "=" * 70)
    print("📊 Test Results")
    print("=" * 70 + "\n")
    print(f"  Response: {response}")
    print(f"  Tokens: {tokens}")
    print(f"  Finish reason: {finish_reason}")
    print(f"\n✅ Test completed")
    print()
