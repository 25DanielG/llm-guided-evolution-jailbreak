import os
import sys
import argparse
import random

# Ensure repo root is on sys.path so `src` imports work from generated bash scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf,
                       llm_code_qc, str2bool, extract_note, generate_augmented_code,
                       clean_code_from_llm, retrieve_base_code)
from llm_mutation import validate_generated_chunk, _trait_assign_name, _mutate_decoding_numerically, _average_decoding
import re


def augment_network(input_filename_x, input_filename_y, output_filename,
                    top_p=0.15, llm_model=LLM_QWEN, temperature=0.1, apply_quality_control=False):
    """Augment Python Network Script.
    
    Parameters
    ----------
    input_filename_x : os.PathLike
        _description_
    input_filename_y : os.PathLike
        _description_
    output_filename : os.PathLike
        _description_
    top_p : float, optional
        _description_, by default 0.15
    llm_model : str, optional
        _description_, by default LLM_QWEN
    temperature : float, optional
        _description_, by default 0.1
    apply_quality_control : bool, optional
        _description_, by default False
    """
    # Split the input files
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    # Create tuples of parts to be augmented
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]))]
    random.shuffle(parts)
    # Find a differing trait segment; if the parents are identical everywhere,
    # fall back to the first trait rather than a stale loop binding.
    x, y, augment_idx = (parts[0] if parts else (parts_x[-1], parts_y[-1], len(parts_x) - 2))
    for cand_x, cand_y, cand_idx in parts:
        if cand_x.strip() != cand_y.strip():
            x, y, augment_idx = cand_x, cand_y, cand_idx
            break
    seg_idx = augment_idx + 1  # actual index into parts_x
    expected_name, _ = _trait_assign_name(x.strip())

    # Numeric decoding trait: combine parents' numbers directly, no LLM call.
    if expected_name == "trait_decoding":
        _, xval = _trait_assign_name(x.strip())
        _, yval = _trait_assign_name(y.strip())
        merged_seed = xval or yval or ""
        code_from_llm = _mutate_decoding_numerically(_average_decoding(xval, yval) if (xval and yval) else merged_seed)
        temp_txt = parts_x[seg_idx]
        note_txt = extract_note(temp_txt)
        parts_x[seg_idx] = f"\n{note_txt}{code_from_llm}\n"
        write_augmented_code(output_filename, parts_x, parts_y)
        box_print(f"Python code saved to {os.path.basename(output_filename)} (numeric decoding crossover, no LLM)",
                  print_bbox_len=120, new_line_end=False)
        print('Job done')
        return

    # Select a template file
    domain_dir = globals().get("CROSSOVER_TEMPLATE_DIR")
    if domain_dir and os.path.isdir(os.path.join(ROOT_DIR, domain_dir)):
        abs_dir = os.path.join(ROOT_DIR, domain_dir)
        template_fname = random.choice(os.listdir(abs_dir))
        template_path = os.path.join(abs_dir, template_fname)
    else:
        template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
        template_path = f'{ROOT_DIR}/templates/CrossOver/{template_fname}'
    with open(template_path, 'r') as file:
        template_txt = file.read()

    # Add code to be augmented
    if template_txt.count("{}") < 2:
        raise ValueError(f"Prompt template {template_path} must contain two literal {{}} placeholders")
    txt2llm = template_txt.replace("{}", x.strip(), 1).replace("{}", y.strip(), 1)
    if "{{FULL_X}}" in txt2llm:
        txt2llm = txt2llm.replace("{{FULL_X}}", "\n\n".join(p.strip() for p in parts_x[1:] if p.strip()))
    if "{{FULL_Y}}" in txt2llm:
        txt2llm = txt2llm.replace("{{FULL_Y}}", "\n\n".join(p.strip() for p in parts_y[1:] if p.strip()))
    # Generate augmented code
    code_from_llm = generate_augmented_code(txt2llm, augment_idx, apply_quality_control,
                                            top_p, llm_model, temperature)

    if not code_from_llm:
        code_from_llm = x.strip()
    else:
        valid_code, invalid_reason = validate_generated_chunk(code_from_llm, expected_name=expected_name)
        if not valid_code:
            print(f"Rejected crossover chunk: {invalid_reason}. Keeping parent segment.", flush=True)
            code_from_llm = x.strip()

    # Insert note if present
    temp_txt = parts_x[seg_idx]
    note_txt = extract_note(temp_txt)
    # Update the part with augmented code
    parts_x[seg_idx] = f"\n{note_txt}{code_from_llm}\n"
    # Prepare and write the augmented code to output file
    write_augmented_code(output_filename, parts_x, parts_y)
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job done')


def write_augmented_code(output_filename, parts_x, parts_y):
    """
    Writes the augmented code to the output file.

    Parameters
    ----------
    output_filename : os.PathLike
        _description_
    parts_x : _type_
        _description_
    parts_y : _type_
        _description_
    """    

    try:
        prompt_log_cross = parts_y[0].split("# --PROMPT LOG--\n")[0]
        prompt_log_cross = f"\n# {'='*10} Start: GeneCrossed\n{prompt_log_cross.strip()}\n# {'='*10} End:\n"
    except IndexError:
        prompt_log_cross = ""

    python_network_txt = prompt_log_cross + '# --OPTION--'.join(parts_x)

    with open(output_filename, 'w') as file:
        file.write(python_network_txt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename_x', type=str, help='Input file name')
    parser.add_argument('input_filename_y', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename_x=args.input_filename_x,
                    input_filename_y=args.input_filename_y,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    llm_model=args.llm_model,
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )
