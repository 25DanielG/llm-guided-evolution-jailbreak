"""Normalize Chat Completions without treating reasoning as an answer."""


class ChatResponseError(RuntimeError):
    def __init__(self, kind, message, data):
        super().__init__(f"Invalid chat completion ({kind}): {message}")
        self.error_type = kind
        choice = (data.get('choices') or [{}])[0] if isinstance(data, dict) else {}
        msg = choice.get('message') or {}
        self.diagnostics = {
            'error_type': kind, 'usage': data.get('usage') or {} if isinstance(data, dict) else {},
            'finish_reason': choice.get('finish_reason'), 'completion_reason': choice.get('stop_reason'),
            'reasoning_present': bool(msg.get('reasoning_content') or msg.get('reasoning')),
            'partial_text': msg.get('content') if isinstance(msg.get('content'), str) else None,
        }


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
            raise ChatResponseError('tool_call', "Tool calls require a tool loop, not final-answer scoring", data)
        if any(marker in content for marker in ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|im_start|>", "<|meta_sep|>", "<|fim_suffix|>")):
            raise ChatResponseError('unparsed_harmony', "Unparsed Harmony in content; check the vLLM runtime/template", data)
        if not content.strip():
            raise ChatResponseError('missing_final_answer', "No final answer; reasoning/token budget exhausted the response", data)
        if choice.get('finish_reason') != 'stop':
            raise ChatResponseError('incomplete_final_answer',
                                    'Final answer did not finish with stop', data)
        return {"text": content, "usage": data.get("usage") or {},
                "reasoning": reasoning, "finish_reason": choice.get("finish_reason"),
                "completion_reason": choice.get('stop_reason'),
                "logprobs": choice.get("logprobs")}
    except (KeyError, IndexError, TypeError, ValueError) as err:
        raise ChatResponseError('malformed_response', str(err), data) from err
