import unittest

from tiny_voca_uz.frontend import numbers as num
from tiny_voca_uz.frontend.normalizer import OKINA, TUTUQ, canonical_apostrophes, cyrillic_to_latin, normalize
from tiny_voca_uz.frontend.phonemizer import NothingToPronounce, phonemize, word_phones
from tiny_voca_uz.symbols import PHONES, SYMBOL_TO_ID


def o(text):
    """Write tests with a plain apostrophe; compare against canonical U+02BB."""
    return text.replace("'", OKINA)


def phones(word):
    return " ".join(word_phones(o(word))[0])


class NumberTests(unittest.TestCase):
    def test_cardinals(self):
        cases = {0: "nol", 7: "yetti", 10: "o'n", 11: "o'n bir", 40: "qirq", 99: "to'qson to'qqiz", 100: "bir yuz",
                 305: "uch yuz besh", 1000: "bir ming", 2026: "ikki ming yigirma olti", 25000: "yigirma besh ming",
                 125000: "bir yuz yigirma besh ming", 1_000_000: "bir million", 3_000_000_000: "uch milliard"}
        for n, words in cases.items():
            self.assertEqual(num.cardinal(n), o(words), n)

    def test_ordinals(self):
        cases = {1: "birinchi", 2: "ikkinchi", 6: "oltinchi", 10: "o'ninchi", 20: "yigirmanchi", 22: "yigirma ikkinchi",
                 40: "qirqinchi", 2026: "ikki ming yigirma oltinchi"}
        for n, words in cases.items():
            self.assertEqual(num.ordinal(n), o(words), n)

    def test_decimals_and_suffixes(self):
        self.assertEqual(num.decimal("3", "5"), o("uch butun o'ndan besh"))
        self.assertEqual(num.decimal("0", "125"), "nol butun mingdan bir yuz yigirma besh")
        self.assertEqual(num.terminative("qirq"), "qirqqacha")
        self.assertEqual(num.terminative("ellik"), "ellikkacha")
        self.assertEqual(num.terminative("besh"), "beshgacha")

    def test_roman(self):
        self.assertEqual(num.roman("XXI"), 21)
        self.assertEqual(num.roman("XIV"), 14)
        with self.assertRaises(ValueError):
            num.roman("IIII")


class NormalizerTests(unittest.TestCase):
    def check(self, cases):
        for text, expected in cases.items():
            self.assertEqual(normalize(text), o(expected), text)

    def test_specification_examples(self):
        self.check({"2026-yil": "ikki ming yigirma oltinchi yil",
                    "25 000 so'm": "yigirma besh ming so'm",
                    "10:30": "soat o'n o'ttiz",
                    "15%": "o'n besh foiz",
                    "3,5": "uch butun o'ndan besh"})

    def test_dates_and_times(self):
        self.check({"2026-yil 22-noyabr": "ikki ming yigirma oltinchi yil yigirma ikkinchi noyabr",
                    "22.11.2026 kuni": "ikki ming yigirma oltinchi yil yigirma ikkinchi noyabr kuni",
                    "Soat 9:05da keling.": "soat to'qqiz nol beshda keling.",
                    "8:00": "soat sakkiz",
                    "2020-2025-yillarda": "ikki ming yigirmanchi — ikki ming yigirma beshinchi yillarda",
                    "XXI asr": "yigirma birinchi asr"})

    def test_quantities(self):
        self.check({"10-15 kishi": "o'ndan o'n beshgacha kishi",
                    "15%ga oshdi": "o'n besh foizga oshdi",
                    "Harorat -5°C": "harorat minus besh daraja",
                    "5 km/soat": "besh kilometr soatiga",
                    "Bu 5 ga yer": "bu besh gektar yer",
                    "5ta olma, 10ga yaqin": "beshta olma, o'nga yaqin",
                    "$100 va 50 €": "bir yuz dollar va ellik yevro",
                    "0,25 litr": "nol butun yuzdan yigirma besh litr",
                    "1 000 000 dollar": "bir million dollar",
                    "5 mln so'm": "besh million so'm"})

    def test_abbreviations(self):
        self.check({"AI, API va TTS": "a i, a pe i va te te es",
                    "AQSh va BMT": "a qe she va be em te",
                    "NATO": "nato",
                    "DIQQAT! YANGI QOIDA": "diqqat! yangi qoida",
                    "olma, nok va h.k.": "olma, nok va hokazo"})

    def test_apostrophes(self):
        for spelling in ("O'zbekiston", "O‘zbekiston", "O’zbekiston", "O`zbekiston", "Oʻzbekiston"):
            self.assertEqual(normalize(spelling), o("o'zbekiston"), spelling)
        self.assertEqual(normalize("ma'no"), "ma" + TUTUQ + "no")
        self.assertEqual(normalize("maʻno"), "ma" + TUTUQ + "no")      # okina used as tutuq
        self.assertEqual(canonical_apostrophes("‘Bugun’"), '"Bugun"')

    def test_cyrillic(self):
        self.assertEqual(cyrillic_to_latin("Ўзбекистон"), o("O'zbekiston"))
        self.assertEqual(cyrillic_to_latin("ШАҲАР"), "SHAHAR")
        self.assertEqual(cyrillic_to_latin("ер, бекат, поезд"), "yer, bekat, poyezd")
        self.assertEqual(cyrillic_to_latin("цирк, милиция"), "sirk, militsiya")
        self.assertEqual(normalize("ғалаба, маъно"), o("g'alaba, ma") + TUTUQ + "no")

    def test_punctuation_and_spacing(self):
        self.check({"u  -  men...": "u — men…", "«Salom»": '"salom"', "Nima?!": "nima?!"})


class PhonemizerTests(unittest.TestCase):
    def test_digraphs_and_special_letters(self):
        self.assertEqual(phones("shahar"), "ʃ æ h æ r")
        self.assertEqual(phones("choy"), "tʃ ɒ j")
        self.assertEqual(phones("ming"), "m i ŋ")
        self.assertEqual(phones("g'alaba"), "ʁ ɑ l æ b æ")
        self.assertEqual(phones("o'g'il"), "o ʁ i l")
        self.assertEqual(phones("is" + TUTUQ + "hoq"), "i s h ɒ q")       # sʼh: s + h, not ʃ
        self.assertEqual(phones("ishoq"), "i ʃ ɒ q")

    def test_q_x_h_are_distinct(self):
        self.assertEqual(phones("qo'l"), "q o l")
        self.assertEqual(phones("xat"), "χ ɑ t")
        self.assertEqual(phones("hol"), "h ɒ l")

    def test_context_rules(self):
        self.assertEqual(phones("kitob"), "k i t ɒ p")          # final devoicing
        self.assertEqual(phones("ozod"), "ɒ z ɒ t")
        self.assertEqual(phones("navro'z"), "n æ w r o z")     # v -> w before a consonant
        self.assertEqual(phones("davom"), "d æ v ɒ m")         # v kept before a vowel
        self.assertEqual(phones("qalam"), "q ɑ l æ m")         # backed a next to q only
        self.assertEqual(phones("ma" + TUTUQ + "no"), "m æ ʔ n ɒ")

    def test_lexicon(self):
        self.assertEqual(phones("jurnal"), "ʒ u r n æ l")
        self.assertEqual(phones("jo'ja"), "dʒ o dʒ æ")

    def test_stress(self):
        result = phonemize("Bugun hozir")
        stressed = [p for p, s in zip(result.symbols, result.stress) if s]
        self.assertEqual(stressed, ["u", "ɒ"])                  # bu'gun (final), ho'zir (lexicon)
        self.assertEqual(sum(phonemize("qo'l").stress), 1)
        self.assertEqual(sum(phonemize("sh").stress), 0)        # no vowel, no stress

    def test_token_layout(self):
        result = phonemize("Bugun havo yaxshi. Ertaga qayerga boramiz?")
        self.assertEqual(result.symbols[0], "<bos>")
        self.assertEqual(result.symbols[-1], "<eos>")
        self.assertEqual(result.symbols.count("<sp>"), 4)
        self.assertIn(".", result.symbols)
        self.assertEqual(result.symbols[-2], "?")
        self.assertNotIn(("<sp>", "."), list(zip(result.symbols, result.symbols[1:])))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(len(result.ids), len(result.stress))
        self.assertEqual(result.unknown, ())

    def test_every_phone_reachable(self):
        text = ("pul bola tosh dala kino gul qo'l ma'no fil vino dav sut zar xat g'oz hol choy jon ts mayda nima "
                "ming lola rang yo'l bir ikki o'g'il jurnal kontsert")
        produced = set(phonemize(text).symbols)
        missing = set(PHONES) - produced - {"ɨ"}       # ɨ is reserved for a later vowel-reduction rule
        self.assertEqual(missing, set())

    def test_full_pipeline_cases(self):
        for text in ("Qishloq xo'jaligi, g'alaba.", "125 000 so'm", "Samarqand, Buxoro, Xorazm",
                     "Kompyuter, internet, texnologiya", "Ўзбекистон Республикаси", "2026-yil 22-noyabr"):
            result = phonemize(text)
            self.assertEqual(result.unknown, (), text)
            self.assertGreater(len(result), 3, text)

    def test_nothing_to_pronounce(self):
        with self.assertRaises(NothingToPronounce):
            phonemize("😀 !!")
        with self.assertRaises(ValueError):
            phonemize("   ")


if __name__ == "__main__":
    unittest.main()
