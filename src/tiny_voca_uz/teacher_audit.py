"""Compare a teacher checkpoint's tensors with the expected 4.5M architecture.

    python -m tiny_voca_uz.teacher_audit path/to/checkpoint.pt > docs/teacher_audit.md

Reports vocabulary size, speaker count, stress levels and channel width read
from the weights, every tensor whose shape differs from ``teacher_config()``
and tensors that exist on only one side. Nothing is executed from the file
beyond ``torch.load(weights_only=True)``.
"""

import argparse
from pathlib import Path
import torch

from .config import teacher_config
from .model import TinyVocaUZ


def state_dict_of(checkpoint):
    for key in ("model", "state_dict", "generator"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def inferred(state):
    shape = lambda key: tuple(state[key].shape) if key in state else None   # noqa: E731
    return {"vocab_size": (shape("tokens.weight") or (None,))[0], "channels": (shape("tokens.weight") or (None, None))[1],
            "speakers": (shape("voices.weight") or (None,))[0], "stress_levels": (shape("stress.weight") or (None,))[0],
            "encoder_layers": len({k.split(".")[2] for k in state if k.startswith("encoder.blocks.")}),
            "prosody_layers": len({k.split(".")[2] for k in state if k.startswith("prosody_stack.blocks.")}),
            "acoustic_layers": len({k.split(".")[2] for k in state if k.startswith("acoustic_stack.blocks.")}),
            "parameters": sum(v.numel() for v in state.values() if torch.is_tensor(v))}


def audit(state, expected=None):
    found = inferred(state)
    expected = expected or teacher_config(vocab_size=found["vocab_size"] or 2, stress_levels=found["stress_levels"] or 1,
                                          speakers=tuple(f"S{i}" for i in range(found["speakers"] or 1)))
    reference = {k: tuple(v.shape) for k, v in TinyVocaUZ(expected).state_dict().items()}
    actual = {k: tuple(v.shape) for k, v in state.items() if torch.is_tensor(v)}
    mismatched = [(k, reference[k], actual[k]) for k in sorted(reference.keys() & actual.keys()) if reference[k] != actual[k]]
    return {"inferred": found, "mismatched": mismatched, "missing": sorted(reference.keys() - actual.keys()),
            "unexpected": sorted(actual.keys() - reference.keys())}


def markdown(result, source):
    lines = ["# Teacher checkpoint audit", "", f"Source: `{source}`", "", "| Property | Value |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |" for k, v in result["inferred"].items()]
    for title, rows in (("Shape mismatches (expected → found)", [f"- `{k}`: {a} → {b}" for k, a, b in result["mismatched"]]),
                        ("Missing from checkpoint", [f"- `{k}`" for k in result["missing"]]),
                        ("Not in the reference architecture", [f"- `{k}`" for k in result["unexpected"]])):
        lines += ["", f"## {title}", ""] + (rows or ["None."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    state = state_dict_of(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    print(markdown(audit(state), args.checkpoint.name), end="")


if __name__ == "__main__":
    main()
