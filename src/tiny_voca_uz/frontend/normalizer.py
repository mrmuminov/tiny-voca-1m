"""Uzbek text normalization: raw text -> lowercase Latin words and punctuation.

Order matters and is fixed:
    1. Unicode cleanup, Cyrillic -> Latin, canonical apostrophes and quotes.
    2. Patterns that contain digits: dates, times, ranges, ordinals, percent,
       currency, units, decimals, signed and plain numbers (with glued suffixes).
    3. Abbreviations and acronyms (case-sensitive, so before lowercasing).
    4. Symbols, lowercasing, whitespace.

Canonical apostrophes: U+02BB (ʻ) in oʻ/gʻ, U+02BC (ʼ) for the tutuq belgisi.
Uzbek writes the decimal separator as a comma and groups thousands with spaces.
"""

import re
import unicodedata

from . import numbers as num

OKINA, TUTUQ = "ʻ", "ʼ"
_APOSTROPHES = "'‘’`´ʻʼ′"

# ---------------------------------------------------------------- Cyrillic

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": TUTUQ, "ь": "", "ы": "i", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + OKINA, "қ": "q", "ғ": "g" + OKINA, "ҳ": "h",
}
_CYRILLIC_VOWELS = set("аеёиоуэюяўы")


def cyrillic_to_latin(text):
    """Official 1995 correspondence, with the context rules for е and ц."""
    out = []
    for i, ch in enumerate(text):
        low = ch.lower()
        prev = text[i - 1].lower() if i else ""
        if low == "е":
            latin = "ye" if not prev.isalpha() or prev in _CYRILLIC_VOWELS or prev in "ъь" else "e"
        elif low == "ц":
            latin = "ts" if prev in _CYRILLIC_VOWELS else "s"
        elif low in _CYRILLIC:
            latin = _CYRILLIC[low]
        else:
            out.append(ch)
            continue
        if ch != low and latin:
            following = text[i + 1] if i + 1 < len(text) else ""
            # ШАХАР -> SHAHAR, Шахар -> Shahar
            latin = latin.upper() if following.isupper() and len(latin) > 1 and latin[1].isalpha() else latin[0].upper() + latin[1:]
        out.append(latin)
    return "".join(out)


def canonical_apostrophes(text):
    """oʻ/gʻ get U+02BB; an apostrophe between letters becomes the tutuq (U+02BC);
    any other apostrophe is a quotation mark."""
    text = re.sub(f"([oOgG])[{_APOSTROPHES}]", lambda m: m.group(1) + OKINA, text)
    text = re.sub(f"(?<=[^\\W\\d_])(?<![oOgG])[{_APOSTROPHES}](?=[^\\W\\d_])", TUTUQ, text)
    return re.sub(f"[{_APOSTROPHES.replace(OKINA, '').replace(TUTUQ, '')}]", '"', text)


def clean(text):
    text = unicodedata.normalize("NFC", text)
    if re.search("[Ѐ-ӿ]", text):
        text = cyrillic_to_latin(text)
    text = canonical_apostrophes(text)
    text = text.translate(str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "–": "—", "―": "—",
                                         " ": " ", " ": " ", " ": " ", "[": "(", "]": ")", "{": "(", "}": ")"}))
    # Letters with foreign diacritics (é, ü, ç) lose them; ʻ and ʼ are kept.
    text = unicodedata.normalize("NFC", "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)))
    return text.replace("...", "…")


# ---------------------------------------------------------------- numbers

UNITS = {
    "km/soat": "kilometr soatiga", "km/s": "kilometr sekundiga", "m/s": "metr sekundiga",
    "km": "kilometr", "m": "metr", "sm": "santimetr", "mm": "millimetr", "kg": "kilogramm", "g": "gramm",
    "t": "tonna", "l": "litr", "ml": "millilitr", "ga": "gektar", "°C": "daraja", "°": "daraja",
    "mln": "million", "mlrd": "milliard", "trln": "trillion",
    "%": "foiz", "$": "dollar", "€": "yevro", "₽": "rubl", "£": "funt",
}
CURRENCY_PREFIX = {"$": "dollar", "€": "yevro", "₽": "rubl", "£": "funt"}
# Symbol units may carry a glued suffix ("15%ga" -> "oʻn besh foizga"). Letter
# units must end at a word boundary, and "ga" (gektar) needs a space, so the
# dative suffix in "5ga" stays a suffix.
_SYMBOL_UNITS = "|".join(re.escape(u) for u in ("°C", "°", "%", "$", "€", "₽", "£"))
_LETTER_UNITS = "|".join(re.escape(u) for u in sorted((u for u in UNITS if u[0].isalpha()), key=len, reverse=True)
                         if u != "ga")
_LETTER = "[^\\W\\d_]"
_NUMBER = r"\d+(?:[.,]\d+)?"


def _read(number):
    if re.fullmatch(r"\d+", number):
        return num.cardinal(int(number))
    whole, fraction = re.split("[.,]", number)
    return num.decimal(whole, fraction)


def _glue(words, suffix):
    """Glue a letter suffix to the last word: (oʻn, ga) -> oʻnga."""
    return words + suffix.lower() if suffix else words


def _date(m):
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    words = f"{num.ordinal(day)} {num.MONTHS[month - 1]}"
    if year:
        words = f"{num.ordinal(int(year))} yil {words}"
    return words


def _time(m):
    hours, minutes, seconds = int(m.group(2)), int(m.group(3)), m.group(4)
    if hours > 24 or minutes > 59 or (seconds and int(seconds) > 59):
        return m.group(0)
    words = [] if m.group(1) else ["soat"]
    words.append(num.cardinal(hours))
    if minutes or seconds:
        words.append(("nol " if m.group(3)[0] == "0" and minutes else "") + num.cardinal(minutes))
    if seconds:
        words.append(num.cardinal(int(seconds)))
    return (m.group(1) or "") + _glue(" ".join(words), m.group(5))


def _ordinal_range(m):
    return f"{num.ordinal(int(m.group(1)))} — {num.ordinal(int(m.group(2)))} {m.group(3)}"


def _ordinal(m):
    return f"{num.ordinal(int(m.group(1)))} {m.group(2)}"


def _range(m):
    return f"{num.ablative(num.cardinal(int(m.group(1))))} {num.terminative(num.cardinal(int(m.group(2))))}"


def _roman_ordinal(m):
    try:
        return f"{num.ordinal(num.roman(m.group(1)))} {m.group(2)}"
    except ValueError:
        return m.group(0)


def _with_unit(m):
    sign, number, unit, suffix = m.group(1), m.group(2), m.group(3) or m.group(4), m.group(m.lastindex)
    words = ("minus " if sign == "-" else "plyus " if sign == "+" else "") + _read(number)
    return _glue(f"{words} {UNITS[unit]}", suffix)


def _currency_prefix(m):
    return _glue(f"{_read(m.group(2))} {CURRENCY_PREFIX[m.group(1)]}", m.group(3))


def _plain(m):
    sign, number, suffix = m.group(1), m.group(2), m.group(3)
    words = ("minus " if sign == "-" else "plyus " if sign == "+" else "") + _read(number)
    return _glue(words, suffix)


def expand_numbers(text):
    # Thousands grouped with spaces: "25 000" -> "25000" (not phone numbers like "90 123 45 67").
    text = re.sub(r"(?<![\d.,])\d{1,3}(?: \d{3})+(?![\d]| \d)", lambda m: m.group(0).replace(" ", ""), text)
    # 22.11.2026 or 22.11 -> "ikki ming yigirma oltinchi yil yigirma ikkinchi noyabr"
    text = re.sub(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?![\d.]\d)", _date, text)
    # 10:30, soat 10:30, 10:30:15, 10:30da
    text = re.sub(rf"((?:[Ss]oat|SOAT)\s+)?(?<![\d:])(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?(?![\d:])({_LETTER}*)",
                  _time, text)
    # 2020-2025-yillar -> ordinals; 10-15 -> oʻndan oʻn beshgacha; 2026-yil -> ordinal
    text = re.sub(rf"(?<![\d-])(\d+)\s*[-—]\s*(\d+)-({_LETTER}+)", _ordinal_range, text)
    text = re.sub(rf"(?<![\d-])(\d+)-({_LETTER}+)", _ordinal, text)
    text = re.sub(r"(?<![\d.,-])(\d{1,4})\s*[-—]\s*(\d{1,4})(?![\d.,-])", _range, text)
    # XXI asr, II bob, IV jahon -> ordinals (only before a lowercase word)
    text = re.sub(r"\b([IVXLCDM]+)\s+(asr|asrda|asrning|bob|qism|bosqich|daraja|jahon|chorak)\b", _roman_ordinal, text)
    # Currency and units after a number; signs only at the start of a token.
    text = re.sub(rf"([$€₽£])\s?({_NUMBER})({_LETTER}*)", _currency_prefix, text)
    text = re.sub(rf"([-+])?(?<![\d.,])({_NUMBER})\s?({_SYMBOL_UNITS})({_LETTER}*)", lambda m: _signed(text, m, _with_unit), text)
    text = re.sub(rf"([-+])?(?<![\d.,])({_NUMBER})(?:\s?({_LETTER_UNITS})|\s(ga))(?![^\W\d_])()",
                  lambda m: _signed(text, m, _with_unit), text)
    text = re.sub(rf"(?:(?<=[\s(\"])|^)([-+])?({_NUMBER})({_LETTER}*)", _plain, text)
    text = re.sub(rf"({_NUMBER})({_LETTER}*)", lambda m: _glue(_read(m.group(1)), m.group(2)), text)
    return text


def _signed(text, m, reader):
    """A leading +/- is a sign only at the start or after whitespace; otherwise
    it is left in place (a hyphen) and only the number is read."""
    if m.group(1) is None or m.start() == 0 or text[m.start() - 1] in " (\"\t\n":
        return reader(m)
    return m.group(1) + reader(_Unsigned(m))


class _Unsigned:
    def __init__(self, m):
        self.m = m
        self.lastindex = m.lastindex

    def group(self, i):
        return None if i == 1 else self.m.group(i)


# ---------------------------------------------------------------- abbreviations

LETTER_NAMES = {
    "A": "a", "B": "be", "C": "se", "D": "de", "E": "e", "F": "ef", "G": "ge", "H": "ha", "I": "i", "J": "je", "K": "ka",
    "L": "el", "M": "em", "N": "en", "O": "o", "P": "pe", "Q": "qe", "R": "er", "S": "es", "T": "te", "U": "u", "V": "ve",
    "W": "ve", "X": "xe", "Y": "ye", "Z": "ze", "O" + OKINA: "o" + OKINA, "G" + OKINA: "g" + OKINA + "e", "SH": "she",
    "CH": "che", "NG": "nge",
}
# Acronyms said as words, and fixed dotted abbreviations. Extend as needed.
ACRONYMS = {"NATO": "nato", "YUNESKO": "yunesko", "YuNESKO": "yunesko", "OAV": "o a ve", "AQSh": "a qe she"}
ABBREVIATIONS = {"h.k.": "hokazo", "va b.": "va boshqalar", "va boshq.": "va boshqalar", "y.": "yil", "mln": "million",
                 "mlrd": "milliard", "trln": "trillion", "sh.": "shahri", "t.y.": "tugʻilgan yili"}
_ACRONYM = re.compile(rf"\b(?:[A-Z]|Sh|Ch|[OG]{OKINA}){{2,7}}\b")


def spell(acronym):
    parts = re.findall(rf"Sh|Ch|SH|CH|[OG]{OKINA}|[A-Z]", acronym)
    return " ".join(LETTER_NAMES[p.upper() if len(p) == 2 and p[1] != OKINA else p] for p in parts)


def expand_abbreviations(text):
    for short, full in ABBREVIATIONS.items():
        text = re.sub(rf"(?<![^\W\d_]){re.escape(short)}(?![^\W\d_])", full, text)
    shouting = not re.search("[a-z]{3}", re.sub(_ACRONYM, "", text))

    def acronym(m):
        word = m.group(0)
        if word in ACRONYMS:
            return ACRONYMS[word]
        if shouting and len(word) > 3:        # an all-caps sentence: ordinary words
            return word
        if sum(ch.isupper() for ch in word) < 2:
            return word
        return spell(word)
    return _ACRONYM.sub(acronym, text)


# ---------------------------------------------------------------- symbols

SYMBOLS = {"&": " va ", "=": " teng ", "№": " raqam ", "#": " raqam ", "@": " kuchukcha ", "+": " plyus ", "×": " karra ",
           "/": " ", "\\": " ", "*": " ", "_": " ", "|": " ", "~": " ", "^": " ", "<": " ", ">": " ",
           "%": " foiz", "°": " daraja"}


def expand_symbols(text):
    text = re.sub(r"\s+-\s+", " — ", text)            # a spaced hyphen is a dash
    text = re.sub(r"(?<![^\W\d_])-|-(?![^\W\d_])", " ", text)   # stray hyphens; word-internal ones stay
    for symbol, word in SYMBOLS.items():
        text = text.replace(symbol, word)
    return text


def normalize(text):
    """Raw Uzbek text (Latin or Cyrillic) -> normalized lowercase Latin."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = clean(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = expand_symbols(text)
    text = text.lower()
    text = re.sub(r"\s+([,.?!;:…)])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
