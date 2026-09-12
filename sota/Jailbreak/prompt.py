"""Seed individual: trait-segmented prompt builder. Only modifies the option marked segments.
Every trait is a function trait_*(state: dict) -> dict that reads and returns shared state.
(keys 'system' and 'user'), applied in TRAIT_ORDER.

Trait axes z = (alpha, pi, phi, kappa, sigma, tau, iota):
- alpha: language / translation wrapper -> trait_language
- pi: persona / framing -> trait_persona
- phi: output format / obfuscation -> trait_format
- kappa: context / scaffolding -> trait_context
- sigma: style / tone -> trait_style
- tau: turn structure -> trait_turns
- iota: instruction phrasing -> trait_instruction
"""

TRAIT_ORDER = [
    "trait_language",
    "trait_persona",
    "trait_format",
    "trait_context",
    "trait_style",
    "trait_turns",
    "trait_instruction",
]

class PromptStrategy:
    """Trait functions into chat prompt"""

    def build_prompt(self, behavior):
        """Return a list of chat messages [{"role", "content"}, ...] for the target."""
        state = {"system": "", "user": str(behavior)}
        for trait_name in TRAIT_ORDER:
            trait_fn = globals().get(trait_name)
            if callable(trait_fn):
                result = trait_fn(state)
                if isinstance(result, dict):
                    state = result
        messages = []
        system_text = state.get("system", "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": state.get("user", "")})
        return messages

# --OPTION--
def trait_language(state):
    # alpha: language / translation wrapper. Seed: identity (English, no wrapper).
    return state
# --OPTION--
def trait_persona(state):
    # pi: persona / framing. Seed: no added persona.
    return state
# --OPTION--
def trait_format(state):
    # phi: output format / obfuscation. Seed: no format constraint.
    return state
# --OPTION--
def trait_context(state):
    # kappa: context / scaffolding. Seed: no added context.
    return state
# --OPTION--
def trait_style(state):
    # sigma: style / tone. Seed: neutral tone.
    return state
# --OPTION--
def trait_turns(state):
    # tau: turn structure. Seed: single-turn (behavior as-is).
    return state
# --OPTION--
def trait_instruction(state):
    # iota: instruction phrasing. Seed: pass the behavior through unchanged.
    return state
