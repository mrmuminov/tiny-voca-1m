"""Uzbek number words: cardinals, ordinals, decimals and fractions.

All output is lowercase Latin with the canonical apostrophes of
``normalizer`` (U+02BB in oʻ/gʻ).
"""

O = "oʻ"   # oʻ with U+02BB

ONES = ("nol", "bir", "ikki", "uch", "toʻrt", "besh", "olti", "yetti", "sakkiz", "toʻqqiz")
TENS = ("", "oʻn", "yigirma", "oʻttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "toʻqson")
SCALES = ((10 ** 12, "trillion"), (10 ** 9, "milliard"), (10 ** 6, "million"), (1000, "ming"))
MONTHS = ("yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr")
MAX = 10 ** 15 - 1
VOWELS = set("aeiouʻ")   # a word ending in "oʻ" ends in a vowel too


def _below_thousand(n):
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += [ONES[hundreds], "yuz"]
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def cardinal(n):
    """1 -> "bir", 2026 -> "ikki ming yigirma olti", 125000 -> "bir yuz yigirma besh ming"."""
    n = int(n)
    if n < 0:
        return "minus " + cardinal(-n)
    if n > MAX:
        return " ".join(ONES[int(d)] for d in str(n))   # read absurdly long numbers digit by digit
    if n == 0:
        return ONES[0]
    words = []
    for value, name in SCALES:
        count, n = divmod(n, value)
        if count:
            words += _below_thousand(count) + [name]
    words += _below_thousand(n)
    return " ".join(words)


def ordinal_suffix(word):
    """bir -> birinchi, ikki -> ikkinchi, yigirma -> yigirmanchi."""
    return word + ("nchi" if word[-1] in VOWELS else "inchi")


def ordinal(n):
    words = cardinal(n).split(" ")
    words[-1] = ordinal_suffix(words[-1])
    return " ".join(words)


def ablative(phrase):
    """Suffix -dan ("from"): oʻn -> oʻndan."""
    return phrase + "dan"


def terminative(phrase):
    """Suffix -gacha ("up to"), assimilated after k/q: qirq -> qirqqacha, ellik -> ellikkacha."""
    last = phrase[-1]
    return phrase + ("qacha" if last == "q" else "kacha" if last == "k" else "gacha")


def denominator(digits):
    """Word for 10**digits used in decimal fractions: 1 -> oʻn, 2 -> yuz, 3 -> ming, 4 -> oʻn ming."""
    return cardinal(10 ** digits).removeprefix("bir ")


def decimal(whole, fraction):
    """("3", "5") -> "uch butun oʻndan besh"; ("0", "25") -> "nol butun yuzdan yigirma besh"."""
    return f"{cardinal(int(whole))} butun {ablative(denominator(len(fraction)))} {cardinal(int(fraction))}"


def digits(text):
    """Read a string of digits one by one (phone numbers, codes)."""
    return " ".join(ONES[int(d)] for d in text)


def roman(text):
    """XXI -> 21. Raises ValueError for anything that is not a canonical numeral."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not text or any(ch not in values for ch in text):
        raise ValueError(text)
    total = 0
    for i, ch in enumerate(text):
        value = values[ch]
        total += -value if i + 1 < len(text) and values[text[i + 1]] > value else value
    if to_roman(total) != text:
        raise ValueError(text)
    return total


def to_roman(n):
    out = []
    for value, symbol in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                          (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        count, n = divmod(n, value)
        out.append(symbol * count)
    return "".join(out)
