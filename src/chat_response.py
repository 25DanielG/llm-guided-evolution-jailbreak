"""Normalize Chat Completions without treating reasoning as an answer."""


def parse_chat_response(data):
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("Expected string message.content")
        if any(marker in content for marker in ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|im_start|>", "<|meta_sep|>", "<|fim_suffix|>")):
            raise ValueError("Unparsed Harmony in content; check the vLLM runtime/template")
        if not content and (reasoning or choice.get("finish_reason") in ("length", "tool_calls")):
            raise ValueError("No final answer; reasoning/token budget or tool call exhausted the response")
        return {"text": content, "usage": data.get("usage") or {},
                "reasoning": reasoning, "finish_reason": choice.get("finish_reason"),
                "logprobs": choice.get("logprobs")}
    except (KeyError, IndexError, TypeError, ValueError) as err:
        raise RuntimeError(f"Invalid chat completion: {err}") from err
