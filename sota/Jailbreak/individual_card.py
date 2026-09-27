"""Individual card exporter: given a gene id, render a readable Markdown page with
its fitness, per-behavior verdicts, and the rendered prompt + target response for
each behavior.

Usage:
  python sota/Jailbreak/individual_card.py <gene_id> [--variant_dir models/llmge_models]

Primary source is results/<gene_id>_results.json (written by eval.py). Else rebuilds prompts from
the evolved module (prompts only, no responses).
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def _fmt_messages(messages):
    if not messages:
        return "_(no messages)_"
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content", "")).strip()
        lines.append(f"**{role}:**\n\n```\n{content}\n```")
    return "\n\n".join(lines)


def _card_from_json(gene_id, data):
    out = []
    out.append(f"# Individual `{gene_id}`\n")
    out.append(f"- mean_judge_score: **{data.get('mean_judge_score')}**")
    out.append(f"- avg_prompt_tokens: {data.get('avg_prompt_tokens')}")
    out.append(f"- behaviors: {data.get('n_jailbroken')}/{data.get('n_behaviors')} jailbroken\n")

    behaviors = data.get("behaviors", [])
    out.append("## Per-behavior summary\n")
    out.append("| # | score | parse_ok | filter | behavior |")
    out.append("|---|-------|----------|--------|----------|")
    for i, b in enumerate(behaviors):
        beh = str(b.get("behavior", "")).replace("|", "\\|")[:80]
        out.append(f"| {i} | {b.get('score')} | {b.get('parse_ok')} | {b.get('filter')} | {beh} |")
    out.append("")

    out.append("## Transcripts\n")
    for i, b in enumerate(behaviors):
        mark = "JAILBROKEN" if (b.get("score") or 0) >= 0.5 else "blocked"
        out.append(f"### [{i}] {mark} — {b.get('behavior','')}\n")
        out.append(_fmt_messages(b.get("messages")))
        resp = b.get("response")
        out.append("\n**target response:**\n\n```\n" + (str(resp).strip() if resp else "(none)") + "\n```\n")
    return "\n".join(out)


def _card_from_module(gene_id, variant_dir):
    """Fallback: rebuild prompts from the evolved module (no responses/verdicts)."""
    import importlib
    variant_abs = os.path.abspath(variant_dir)
    sys.path.insert(0, variant_abs)
    os.chdir(SCRIPT_DIR)
    module_name = f"prompt_{gene_id}"
    strategy = importlib.import_module(module_name).PromptStrategy()
    import eval as ev
    behaviors = ev.load_behaviors()
    out = [f"# Individual `{gene_id}` (prompts only — no eval results found)\n"]
    decoding_shown = False
    for i, beh in enumerate(behaviors):
        try:
            messages, decoding = ev._unpack_build_prompt(strategy.build_prompt(beh))
        except Exception as err:
            messages, decoding = None, {}
            note = f"_(build_prompt failed: {err})_"
        else:
            note = ""
        if decoding and not decoding_shown:
            out.append(f"_decoding overrides: {decoding}_\n")
            decoding_shown = True
        out.append(f"### [{i}] {beh}\n{note}\n")
        out.append(_fmt_messages(messages if isinstance(messages, list) else None))
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_id")
    ap.add_argument("--variant_dir", default="models/llmge_models")
    ap.add_argument("--out", default=None, help="output .md path (default results/<gene_id>_card.md)")
    args = ap.parse_args()

    json_path = os.path.join(SCRIPT_DIR, "results", f"{args.gene_id}_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            card = _card_from_json(args.gene_id, json.load(f))
    else:
        print(f"no {json_path}; rebuilding prompts from module", flush=True)
        card = _card_from_module(args.gene_id, args.variant_dir)

    out_path = args.out or os.path.join(SCRIPT_DIR, "results", f"{args.gene_id}_card.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(card)
    print(f"card written to {out_path}")


if __name__ == "__main__":
    main()
