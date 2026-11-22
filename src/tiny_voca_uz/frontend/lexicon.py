"""Exceptions to the letter-to-sound and stress rules.

PRONUNCIATIONS: whole-word phoneme strings (space separated, ˈ before the
stressed vowel). Use for loanwords whose spelling the rules misread.

STRESS: words whose stress is not on the last syllable, as the 0-based index
of the stressed syllable counted from the start (negative counts from the end).
Most Uzbek words are stressed on the final syllable; the exceptions below are
common adverbs, conjunctions and question words.

Both tables are lowercase, normalized spelling (U+02BB in oʻ/gʻ).
"""

PRONUNCIATIONS = {
    # ʒ in loanwords (native j is dʒ)
    "jurnal": "ʒ u r n ˈæ l", "jurnalist": "ʒ u r n æ l ˈi s t", "janr": "ʒ ˈæ n r", "jaket": "ʒ æ k ˈe t",
    "garaj": "ɡ æ r ˈæ ʒ", "massaj": "m æ s s ˈæ ʒ", "tiraj": "t i r ˈæ ʒ",
    "etaj": "e t ˈæ ʒ", "reportaj": "r e p ɒ r t ˈæ ʒ", "jyuri": "ʒ u r ˈi",
    # ts in loanwords
    "kontsert": "k ɒ n ts ˈe r t", "sirk": "ts ˈi r k",
    # n+g across a morpheme boundary, not ŋ
    "ungacha": "u n ɡ æ tʃ ˈæ", "bungacha": "b u n ɡ æ tʃ ˈæ", "shungacha": "ʃ u n ɡ æ tʃ ˈæ",
}

# Only well-attested cases; to be reviewed and extended by a native speaker.
STRESS = {
    "hozir": 0, "lekin": 0, "faqat": 0, "balki": 0, "chunki": 0, "hatto": 0, "ammo": 0, "albatta": 1, "hamma": 0,
    "nima": 0, "nega": 0, "qani": 0, "mana": 0, "ana": 0, "zero": 0, "baʼzan": 0,
}
