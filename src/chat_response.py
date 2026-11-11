"""Normalize Chat Completions without treating reasoning as an answer."""


def parse_chat_response(data):
    try:
        choice = data["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise ValueError("Expected message object")
        content = message.get("content")
        # vLLM 0.17 uses reasoning; retain compatibility with older endpoints.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("Expected string message.content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("Expected string reasoning")
        if message.get("tool_calls") or message.get("function_call") or choice.get("finish_reason") in ("tool_calls", "function_call"):
            raise ValueError("Tool calls require a tool loop, not final-answer scoring")
        if any(marker in content for marker in ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|im_start|>", "<|meta_sep|>", "<|fim_suffix|>")):
            raise ValueError("Unparsed Harmony in content; check the vLLM runtime/template")
        if not content.strip():
            raise ValueError("No final answer; reasoning/token budget or tool call exhausted the response")
        return {"text": content, "usage": data.get("usage") or {},
                "reasoning": reasoning, "finish_reason": choice.get("finish_reason"),
                "logprobs": choice.get("logprobs")}
    except (KeyError, IndexError, TypeError, ValueError) as err:
        raise RuntimeError(f"Invalid chat completion: {err}") from err
