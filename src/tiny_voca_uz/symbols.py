"""Minimal Uzbek (Latin script) phoneme inventory, front-end version 0.

This is only the token table the model is built around. The normalizer and
the grapheme-to-phoneme rules come in the front-end stage. If the teacher
checkpoint uses a different inventory, the student adopts the teacher's so
that token ids line up for distillation (see docs/teacher_audit.md).

Append-only once a checkpoint exists: reordering invalidates trained weights.
"""

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "(", ")", "—", "…")

# Grapheme -> phoneme, for reference:
#   sh ʃ, ch tʃ, j dʒ (also ʒ in loans), g‘ ʁ, ng ŋ, q q, x χ, h h, y j,
#   tutuq belgisi (’) ʔ, v v/w, ts (loanwords: sirk, kontsert).
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "q", "ʔ", "f", "v", "w", "s", "z", "ʃ", "ʒ",
              "χ", "ʁ", "h", "tʃ", "dʒ", "ts", "m", "n", "ŋ", "l", "r", "j")
# i, e, a (front æ / back ɑ allophones), o (ɒ), u, o‘ (o). ɨ/ɪ is the reduced i
# heard in closed unstressed syllables (e.g. "bilan").
VOWELS = ("i", "ɨ", "e", "æ", "ɑ", "ɒ", "o", "u")
PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES

SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}

# Stress is a parallel feature: 0 unstressed, 1 stressed. Uzbek has no
# secondary lexical stress, so the original 3-way table shrinks to 2.
STRESS_LEVELS = 2

if len(set(SYMBOLS)) != len(SYMBOLS):
    raise RuntimeError("duplicate symbol in the inventory")
