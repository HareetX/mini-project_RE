"""Simple OpenAI client with retries and batch helpers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Mapping, Sequence

from openai import OpenAI, OpenAIError

import configs


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        timeout: int = 30,
        max_retries: int = configs.MAX_RETRIES,
        backoff_factor: float = 2.0,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key or configs.API_KEY,
            base_url=base_url or configs.BASE_URL,
            timeout=timeout,
        )
        self.model = model or configs.MODEL_NAME
        self.embedding_model = embedding_model or configs.EMBEDDINGS_MODEL_NAME
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except OpenAIError:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= self.backoff_factor

    def generate(
        self,
        messages: Sequence[Mapping[str, str]] | str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self._retry(
            self.client.chat.completions.create,
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_batch(
        self,
        prompts: Sequence[str] | Sequence[Sequence[Mapping[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        parallelism: int = 4,
    ) -> List[str]:
        results: list[str | None] = [None] * len(prompts)

        def _task(idx: int, item: Any) -> None:
            results[idx] = self.generate(item, temperature=temperature, max_tokens=max_tokens)

        parallelism = min(parallelism, len(prompts))
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(_task, i, prompt) for i, prompt in enumerate(prompts)]
            for future in as_completed(futures):
                future.result()
        return [text or "" for text in results]

    def embedding(self, text: str) -> List[float]:
        response = self._retry(
            self.client.embeddings.create,
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def embedding_batch(
        self,
        texts: Iterable[str],
        parallelism: int = 4,
    ) -> List[List[float]]:
        texts_list = list(texts)
        results: list[List[float] | None] = [None] * len(texts_list)

        def _task(idx: int, item: str) -> None:
            results[idx] = self.embedding(item)

        parallelism = min(parallelism, len(texts_list))
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(_task, i, text) for i, text in enumerate(texts_list)]
            for future in as_completed(futures):
                future.result()
        return [vec or [] for vec in results]
    
    def parse_json(self, response_text: str) -> Any:
        import json

        # Attempt to find the first and last curly braces
        json_type = "dict" # default to dict
        start_idx_dict = response_text.find("{")
        start_idx_list = response_text.find("[")
        if start_idx_dict == -1 and start_idx_list == -1:
            raise ValueError("No valid JSON object found in response text")
        elif start_idx_dict == -1:
            start_idx = start_idx_list
            json_type = "list"
        elif start_idx_list == -1:
            start_idx = start_idx_dict
        else:
            start_idx = min(start_idx_dict, start_idx_list)
            if start_idx == start_idx_list:
                json_type = "list"
        
        end_idx = response_text.rfind("}")
        if json_type == "list":
            end_idx = response_text.rfind("]")

        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise ValueError("No valid JSON object found in response text")

        json_str = response_text[start_idx:end_idx + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response text: {json_str}") from e


__all__ = ["OpenAIClient"]


if __name__ == "__main__":
    # Simple test
    client = OpenAIClient()

    # Test generate
    print(client.generate("Hello, how are you?"))

    # Test embedding
    print(client.embedding("Hello, how are you?"))

    # Test parse_json
    test_response = '''Here is the data you requested:
{"key1": "value1", "key2": 2, "key3": [1, 2, 3]}'''
    print(client.parse_json(test_response))
    test_response_list = '''The results are as follows:
[{"item": "A"}, {"item": "B"}, {"item": "C"}]'''
    print(client.parse_json(test_response_list))