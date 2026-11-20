"""CPU-only serving capacity calculations and full chat-context checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

GIB = 1024**3


def literal_traits(source):
    """Read all prompt traits, including system/instruction text, without exec."""
    traits = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith('trait_'):
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    raise ValueError('Nonliteral prompt trait: ' + target.id)
                traits[target.id] = node.value.value
    return traits


def kv_upper_bound(config, context, sequences, tp, element_bytes=2):
    """Conservative all-layers/full-attention KV bytes per TP rank (no padding)."""
    heads = config['num_key_value_heads']
    if min(context,sequences,tp,element_bytes) <= 0 or heads % tp:
        raise ValueError('Invalid KV dimensions or nondivisible KV heads')
    return (2 * config['num_hidden_layers'] * (heads // tp) * config['head_dim']
            * element_bytes * context * sequences)


def require_context(prompt_tokens, output_tokens, max_model_len):
    if min(prompt_tokens,output_tokens,max_model_len) <= 0:
        raise ValueError('Token counts must be positive')
    if prompt_tokens + output_tokens > max_model_len:
        raise ValueError(f'Full prompt plus output exceeds context: {prompt_tokens}+{output_tokens}>{max_model_len}')


def require_context_audit(config, folder):
    path = Path(folder)/'serving-context-audit.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['context_audit_sha256']:
        raise ValueError('Context audit changed')
    audit = json.loads(path.read_text())
    if (audit['status'] != 'passed' or audit['dataset_sha256'] != config['dataset_sha256']
            or audit['tokenizer_revision'] != config['model_profile']['revision']
            or audit['output_tokens'] != 512
            or audit['max_model_len'] != config['model_profile']['max_model_len']):
        raise ValueError('Context audit differs from experiment')
    require_context(audit['max_prompt_tokens'],512,config['model_profile']['max_model_len'])


def audit_context(tokenizer, renderer, trait_sources, behaviors, output_tokens, max_model_len):
    """Inspect literal traits, never execute generated gene source or truncate it."""
    counts, unique, longest, files = [], set(), None, []
    for filename in trait_sources:
        path = Path(filename); source = path.read_text()
        files.append(dict(path=str(path),sha256=hashlib.sha256(source.encode()).hexdigest()))
        traits = literal_traits(source)
        key = json.dumps(traits,sort_keys=True)
        if key in unique:
            continue
        unique.add(key)
        for behavior in behaviors:
            messages,_ = renderer(traits,behavior)
            count = len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
            require_context(count,output_tokens,max_model_len)
            counts.append(count)
            if longest is None or count > longest['prompt_tokens']:
                longest = dict(path=str(path),prompt_tokens=count,messages=messages)
    if not counts:
        raise ValueError('No rendered prompts audited')
    return dict(status='passed',source_files=files,unique_trait_sets=len(unique),
                behaviors=len(behaviors),rendered_prompts=len(counts),
                min_prompt_tokens=min(counts),max_prompt_tokens=max(counts),
                p95_prompt_tokens=sorted(counts)[int(.95*(len(counts)-1))],
                output_tokens=output_tokens,max_model_len=max_model_len,
                required_context=max(counts)+output_tokens,longest=longest)


def stress_messages(tokenizer, prompt_budget, marker):
    """Benign unique-prefix input within eight tokens of the full input budget."""
    def build(n):
        return [{'role':'user','content':f'Resource test {marker}. Background vocabulary: '
                 + ' molecule'*n + '. Explain photosynthesis in two sentences.'}]
    low, high = 0, prompt_budget*2
    while low < high:
        mid=(low+high+1)//2
        count=len(tokenizer.apply_chat_template(build(mid),tokenize=True,add_generation_prompt=True))
        if count <= prompt_budget: low=mid
        else: high=mid-1
    messages=build(low)
    count=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    if not prompt_budget-8 <= count <= prompt_budget:
        raise ValueError('Cannot construct an input near the full context limit')
    return messages,count
