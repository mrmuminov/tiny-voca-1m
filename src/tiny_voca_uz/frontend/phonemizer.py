"""Uzbek Latin -> phoneme tokens with a parallel stress feature.

Token layout follows the original tiny-voca front end so the model code is
unchanged: <bos>, phonemes of each word, <sp> between two words, punctuation
tokens where the text has them (no <sp> next to punctuation), <eos>. Stress
(0/1) sits on the vowel of the stressed syllable.

Letter-to-sound rules (normalized spelling in, ``symbols.PHONES`` out):
    sh ʃ   ch tʃ   ng ŋ   gʻ ʁ   oʻ o   o ɒ   x χ   j dʒ   y j   q q   h h
    tutuq (ʼ) ʔ, except in sʼh where it only separates s and h
    c s before e/i/y, k elsewhere;  w v
Context rules, applied after the letter rules:
    a   ɑ next to q, ʁ or χ, æ elsewhere
    v   w after a vowel when no vowel follows (navroʻz, dav)
    b d g at the end of a word are devoiced to p t k (kitob -> kitɒp)
Stress: last syllable, unless ``lexicon.STRESS`` or ``lexicon.PRONUNCIATIONS``
says otherwise. Suffix-driven exceptions (unstressed -mi, -man, ...) need a
morphological analyser and are not handled yet.
"""

from dataclasses import dataclass
import re

from ..symbols import BOS, EOS, PHONES, PUNCTUATION, SP, SYMBOL_TO_ID, UNK
from . import lexicon
from .normalizer import OKINA, TUTUQ, normalize

VOWELS = frozenset(("i", "ɨ", "e", "æ", "ɑ", "ɒ", "o", "u"))
UVULARS = frozenset(("q", "ʁ", "χ"))
DEVOICE = {"b": "p", "d": "t", "ɡ": "k"}
LETTERS = {"a": "æ", "b": "b", "d": "d", "e": "e", "f": "f", "g": "ɡ", "h": "h", "i": "i", "j": "dʒ", "k": "k",
           "l": "l", "m": "m", "n": "n", "o": "ɒ", "p": "p", "q": "q", "r": "r", "s": "s", "t": "t", "u": "u", "v": "v",
           "x": "χ", "y": "j", "z": "z", "w": "v"}
DIGRAPHS = {"sh": "ʃ", "ch": "tʃ", "ng": "ŋ", "g" + OKINA: "ʁ", "o" + OKINA: "o"}
_WORD = re.compile(f"[a-z{OKINA}{TUTUQ}]+|[^\\sa-z{OKINA}{TUTUQ}]")


class NothingToPronounce(ValueError):
    """The text has no word to say (only marks, emoji and the like)."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple      # characters that had no phoneme

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return " ".join(("ˈ" if s else "") + p for p, s in zip(self.symbols, self.stress))


def letters_to_phones(word):
    """Letter rules only. Returns (phones, unknown characters)."""
    phones, unknown, i = [], [], 0
    while i < len(word):
        pair = word[i:i + 2]
        if word[i:i + 3] == "s" + TUTUQ + "h":
            phones += ["s", "h"]
            i += 3
        elif pair in DIGRAPHS:
            phones.append(DIGRAPHS[pair])
            i += 2
        elif word[i] == TUTUQ:
            if phones:                       # word-initial tutuq has nothing to stop
                phones.append("ʔ")
            i += 1
        elif word[i] == OKINA:               # stray okina: no sound
            i += 1
        elif word[i] == "c":
            phones.append("s" if word[i + 1:i + 2] in ("e", "i", "y") else "k")
            i += 1
        elif word[i] in LETTERS:
            phones.append(LETTERS[word[i]])
            i += 1
        else:
            unknown.append(word[i])
            phones.append(UNK)
            i += 1
    return phones, unknown


def apply_context(phones):
    out = list(phones)
    for i, p in enumerate(out):
        before = out[i - 1] if i else None
        after = out[i + 1] if i + 1 < len(out) else None
        if p == "æ" and (before in UVULARS or after in UVULARS):
            out[i] = "ɑ"
        elif p == "v" and before in VOWELS and after not in VOWELS:
            out[i] = "w"
    if out and out[-1] in DEVOICE:
        out[-1] = DEVOICE[out[-1]]
    return out


def stressed_vowel(phones, word):
    vowels = [i for i, p in enumerate(phones) if p in VOWELS]
    if not vowels:
        return None
    index = lexicon.STRESS.get(word, -1)
    return vowels[index] if -len(vowels) <= index < len(vowels) else vowels[-1]


def word_phones(word):
    """One normalized word -> (phones, stress flags, unknown characters)."""
    if word in lexicon.PRONUNCIATIONS:
        phones, stress = [], []
        for item in lexicon.PRONUNCIATIONS[word].split():
            phones.append(item.lstrip("ˈ"))
            stress.append(int(item.startswith("ˈ")))
        return phones, stress, []
    phones, unknown = letters_to_phones(word)
    phones = apply_context(phones)
    target = stressed_vowel(phones, word)
    return phones, [int(i == target) for i in range(len(phones))], unknown


def tokenize(normalized):
    symbols, stress, unknown = [BOS], [0], []
    for piece in _WORD.findall(normalized):
        if piece in PUNCTUATION:
            if symbols[-1] != piece:           # collapse repeats such as "!!"
                symbols.append(piece)
                stress.append(0)
            continue
        if not piece[0].isalpha() and piece[0] not in (OKINA, TUTUQ):
            if piece == "-" or piece.isspace():
                continue
            unknown.append(piece)
            continue
        phones, flags, missing = word_phones(piece)
        if not phones:
            continue
        unknown += missing
        if symbols[-1] in PHONES or symbols[-1] == UNK:
            symbols.append(SP)
            stress.append(0)
        symbols += phones
        stress += flags
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONES or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    """Raw text -> tokens. Runs the normalizer first."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    return tokenize(normalize(text))
