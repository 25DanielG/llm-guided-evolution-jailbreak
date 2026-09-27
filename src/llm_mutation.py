import os
import sys
import re
import ast
import time
import glob
import numpy as np
import transformers
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work even when launched from nested dirs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print

from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, generate_augmented_code, 
                       extract_note, clean_code_from_llm, retrieve_base_code)

def _trait_assign_name(code_from_llm):
    """If code_from_llm is exactly one `trait_x = "..."` assignment, return
    (name, string_value); else (None, None)"""
    try:
        tree = ast.parse(code_from_llm)
    except SyntaxError:
        return None, None
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
    trait_assigns = [
        (t.id, n) for n in assigns for t in n.targets
        if isinstance(t, ast.Name) and t.id.startswith("trait_")
    ]
    if len(trait_assigns) != 1:
        return None, None
    name, node = trait_assigns[0]
    value = node.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return name, value.value
    return name, None

def validate_generated_chunk(code_from_llm, expected_name=None):
    """Reject invalid generated segments.

    1. Forbidden substring patterns from config (all domains).
    2. Structural trait validation.
    Returns (ok, reason).
    """
    try:
        for pattern, reason in FORBIDDEN_PATTERNS:
            if pattern in code_from_llm:
                return False, reason
    except NameError:
        # FORBIDDEN_PATTERNS not defined for this domain; skip substring check.
        pass

    if globals().get("TRAIT_VALIDATION") != "jailbreak_tags":
        return True, ""

    name, value = _trait_assign_name(code_from_llm)
    if name is None:
        return False, "chunk must be exactly one `trait_x = \"...\"` string assignment"
    if value is None:
        return False, f"{name}'s value must be a plain string literal (tag text), not an expression"
    if expected_name is not None and name != expected_name:
        return False, f"expected assignment name {expected_name!r}, got {name!r} (would break position mapping)"
    expected_tag = name[len("trait_"):] if name.startswith("trait_") else name
    if f"<{expected_tag}" not in value:
        return False, f"{name}'s value must contain a <{expected_tag}> tag"
    return True, ""

_DECODING_RE = re.compile(r'temperature="([^"]*)"|top_p="([^"]*)"')

def _mutate_decoding_numerically(code2llm):
    """trait_decoding is purely numeric -- perturb temperature/top_p directly in
    Python (no LLM call), per IMPROVE.md's '8th trait, no LLM involved'."""
    temp_m = re.search(r'temperature="([^"]*)"', code2llm)
    top_p_m = re.search(r'top_p="([^"]*)"', code2llm)
    try:
        temp = float(temp_m.group(1)) if temp_m else 0.7
    except ValueError:
        temp = 0.7
    try:
        top_p = float(top_p_m.group(1)) if top_p_m else 1.0
    except ValueError:
        top_p = 1.0
    temp = max(0.0, min(2.0, temp + np.random.uniform(-0.3, 0.3)))
    top_p = max(0.05, min(1.0, top_p + np.random.uniform(-0.15, 0.15)))
    return f'trait_decoding = \'<decoding temperature="{temp:.2f}" top_p="{top_p:.2f}"/>\''

def _average_decoding(xval, yval):
    """Blend two trait_decoding tag values (crossover), before the usual
    numeric jitter is applied on top."""
    def parse(v):
        tm = re.search(r'temperature="([^"]*)"', v or "")
        pm = re.search(r'top_p="([^"]*)"', v or "")
        try:
            t = float(tm.group(1)) if tm else 0.7
        except ValueError:
            t = 0.7
        try:
            p = float(pm.group(1)) if pm else 1.0
        except ValueError:
            p = 1.0
        return t, p
    tx, px = parse(xval)
    ty, py = parse(yval)
    return f'trait_decoding = \'<decoding temperature="{(tx+ty)/2:.2f}" top_p="{(px+py)/2:.2f}"/>\''

def _read_feedback_summary(input_filename):
    try:
        variant_dir = os.path.dirname(os.path.abspath(input_filename))
        base = os.path.splitext(os.path.basename(input_filename))[0]  # {MODEL}_{gene_id}
        gene_id = base.split("_", 1)[1] if "_" in base else base
        sota_root = globals().get("SOTA_ROOT")
        if not sota_root:
            return ""
        results_path = os.path.join(sota_root, "results", f"{gene_id}_results.json")
        if not os.path.exists(results_path):
            return ""
        import json
        with open(results_path) as f:
            data = json.load(f)
        behaviors = data.get("behaviors", [])
        succeeded = [b["behavior"] for b in behaviors if b.get("harm")][:3]
        failed = [b["behavior"] for b in behaviors if not b.get("harm")][:3]
        lines = [f"This individual scored harm_success_rate={data.get('harm_success_rate')} last time."]
        if succeeded:
            lines.append("It already succeeds on asks like: " + "; ".join(s[:80] for s in succeeded))
        if failed:
            lines.append("It still fails on asks like: " + "; ".join(s[:80] for s in failed) +
                          " -- try to generalize toward these.")
        return "\n".join(lines)
    except Exception:
        return ""

def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, llm_model=LLM_DEEPSEEK, temperature=0.1, apply_quality_control=False):

    print(f'Loading {input_filename} code')
    print('Using')
    parts = split_file(input_filename)
    augment_idx = np.random.randint(1, len(parts))
    # select code to be augmented randomly
    code2llm = parts[augment_idx]
    expected_name, _ = _trait_assign_name(code2llm.strip())

    if expected_name == "trait_decoding":
        code_from_llm = _mutate_decoding_numerically(code2llm)
        note_txt = extract_note(code2llm)
        parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
        python_network_txt = '# --OPTION--'.join(parts)
        output_file = Path(output_filename)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_text(python_network_txt)
        box_print(f"Python code saved to {os.path.basename(output_filename)} (numeric decoding mutation, no LLM)",
                  print_bbox_len=120, new_line_end=False)
        print('Job Done')
        return

    fname = os.path.join(ROOT_DIR, template_txt)
    with open(fname, 'r') as file:
        template_txt = file.read()

    # Prompt templates use a single literal "{}" marker for the code chunk.
    # Using str.format would treat JSON/dict examples in the prompt as fields.
    if "{}" not in template_txt:
        raise ValueError(f"Prompt template {fname} must contain a literal {{}} code placeholder")
    txt2llm = template_txt.replace("{}", code2llm.strip(), 1)
    if "{{FULL}}" in txt2llm:
        full_doc = "\n\n".join(p.strip() for p in parts[1:] if p.strip())
        txt2llm = txt2llm.replace("{{FULL}}", full_doc)
    if "{{FEEDBACK}}" in txt2llm:
        txt2llm = txt2llm.replace("{{FEEDBACK}}", _read_feedback_summary(input_filename) or "(no prior eval yet)")
    code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                            top_p, llm_model, temperature)

    if not code_from_llm:
        code_from_llm = code2llm.strip()
    else:
        valid_code, invalid_reason = validate_generated_chunk(code_from_llm, expected_name=expected_name)
        if not valid_code:
            print(f"Rejected generated chunk: {invalid_reason}. Falling back to parent chunk.", flush=True)
            code_from_llm = code2llm.strip()

    note_txt = extract_note(code2llm)
    parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    # prompt_log = f'# Parent Prompt: {template_path} Root Code: {input_filename}\n'
    # python_network_txt = prompt_log + '# --OPTION--'.join(parts)
    python_network_txt = '# --OPTION--'.join(parts)
    # Write the text to the file
    output_file = Path(output_filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(python_network_txt)


    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job Done')


    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')

    # Parse the arguments
    args = parser.parse_args()
    

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    llm_model=args.llm_model,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )
