"""Serialize existing mutation instructions for the configured tokenizer."""


def prepare_mutator_inputs(tokenizer, prompts):
    if not getattr(tokenizer, 'chat_template', None):
        return prompts, {}
    rendered = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts]
    # The chat template already contains its special tokens (including BOS).
    return rendered, {'add_special_tokens': False}
