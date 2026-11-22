"""python -m tiny_voca_uz.frontend "Matn" [...]  — print normalized text and phonemes."""

import sys

from .normalizer import normalize
from .phonemizer import phonemize


def main(argv=None):
    texts = (argv if argv is not None else sys.argv[1:]) or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    for text in texts:
        result = phonemize(text)
        print(f"text:       {text}\nnormalized: {normalize(text)}\nphonemes:   {result}")
        if result.unknown:
            print(f"unknown:    {' '.join(result.unknown)}")
        print()


if __name__ == "__main__":
    main()
