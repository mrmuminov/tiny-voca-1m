# Tiny-Voca-UZ-1M

Taxminan 1 million parametrli, bitta tilga (o‘zbek tili) va bitta ovozga ixtisoslashgan TTS model.
24 kHz, CPU inference, keyinchalik ONNX. Arxitektura [tiny-voca](https://github.com/n4ze3m/tiny-voca) (Apache-2.0)
asosida: FastSpeech 2 uslubidagi transformer + source-filter synthesizer (neural vocoder yo‘q).

> Holat: **1-deliverable** (audit, parametr hisobi, 1M student, forward testlar) va
> **2-bosqich boshlandi**: o‘zbek normalizer va phonemizer. Dataset, distillation va training hali yo‘q.

## Parametrlar (o‘lchangan, `docs/parameter_audit.md`)

| Modul | Original (EN) | Teacher arxitekturasi (UZ inventar) | **Student 1M** |
|---|---:|---:|---:|
| Embeddings | 16,704 | 11,328 | 5,664 |
| Text encoder | 1,588,864 | 1,588,864 | 338,016 |
| Duration predictor | 123,906 | 123,906 | 56,066 |
| Prosody stack A + head | 795,971 | 795,971 | 226,083 |
| Acoustic stack B | 1,985,984 | 1,985,984 | 338,016 |
| Source-filter head | 25,862 | 25,862 | 12,998 |
| **Jami** | **4,537,291** | **4,531,915** | **976,843** |

Original modelning 4,537,291 parametri kodda aynan qayta hosil qilinadi (test bilan tekshiriladi).
Student hajmi: FP32 3.9 MB, FP16 1.95 MB, INT8 ~1 MB (faqat weight'lar).

### Spetsifikatsiyaga tuzatish: hidden 80 emas, 96

Hidden=80 va 3/1/2 qatlam bilan model atigi **464,715** parametr bo‘ladi — 1M budjetning yarmi ishlatilmaydi.
Tanlangan konfiguratsiya:

| | Original | Student |
|---|---:|---:|
| hidden (channels) | 192 | **96** |
| heads | 4 | 4 (head width 24) |
| feedforward | 640 | 384 |
| encoder / prosody / acoustic bloklar | 4 / 2 / 5 | **3 / 2 / 3** |
| predictor channels | 128 | 96 |
| speakers | 2 | 1 |
| stress darajalari | 3 | 2 |

96 = teacher kengligining aynan yarmi, shuning uchun feature distillation uchun faqat 96→192 linear projection
kerak bo‘ladi (u training-only, 1M hisobiga kirmaydi). Boshqa variantlar (80…128 kenglik, turli chuqurlik)
`docs/parameter_audit.md` dagi sweep jadvalida — ablation uchun.

## Tuzilma

```
src/tiny_voca_uz/
  symbols.py        o‘zbekcha fonema inventari (51 token; frontend v0, faqat jadval)
  config.py         Config (student defaultlari), teacher_config(), english_teacher_config()
  model.py          TinyVocaUZ — original model.py dan olingan; return_hidden distillation uchun
  params.py         modul bo‘yicha parametr hisobi, tensor shape'lar, sweep
  teacher_audit.py  teacher checkpoint state_dict'ini kutilgan arxitektura bilan solishtirish
  frontend/
    normalizer.py   matn → normallashgan lotin (Kirill, apostrof, raqam, sana, vaqt, birlik, qisqartma)
    numbers.py      sonlar: miqdor, tartib, kasr, -dan/-gacha, rim raqamlari
    phonemizer.py   lotin → fonema tokenlari + urg‘u
    lexicon.py      talaffuz va urg‘u istisnolari
configs/model_1m.json
docs/parameter_audit.md
tests/test_model.py
```

## Ishga tushirish

```bash
pip install torch numpy           # CPU yetarli
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m tiny_voca_uz.params --config configs/model_1m.json
PYTHONPATH=src python -m tiny_voca_uz.params --markdown > docs/parameter_audit.md
```

Testlar: original parametr soni (4,537,291), student 950K–1.05M oralig‘ida, teacher-forced forward
(`encode → expand → prosody → decode → vocoder`), free-running `synthesize()`, batch padding audioni o‘zgartirmasligi,
teacher audit'ning original layout'ni to‘g‘ri o‘qishi.

## O‘zbek frontend

```bash
PYTHONPATH=src python -m tiny_voca_uz.frontend "2026-yil 22-noyabr soat 10:30 da 15% chegirma."
```

**Normalizer** (tartib qat'iy):
1. Unicode tozalash; Kirill → Lotin (е/ц kontekst qoidalari bilan); apostroflar: `oʻ gʻ` → U+02BB,
   so‘z ichidagi tutuq belgisi → U+02BC, qolganlari qo‘shtirnoq.
2. Raqamli naqshlar: sana (`22.11.2026`), vaqt (`10:30` → "soat o‘n o‘ttiz", oldida "soat" bo‘lsa takrorlanmaydi),
   `N-so‘z` → tartib son, `10-15` → "o‘ndan o‘n beshgacha", foiz, valyuta, birliklar (km, kg, °C …), kasr (`3,5`),
   ishora (`-5`), minglik guruhlar (`25 000`), yopishgan qo‘shimchalar (`5ta`, `15%ga`), rim raqami + asr/bob.
3. Qisqartmalar: `h.k.`, `mln`; harfma-harf o‘qiladigan akronimlar (`TTS` → "te te es", `AQSh` → "a qe she");
   to‘liq katta harfli gaplar akronim deb o‘qilmaydi.
4. Belgilar (`&`, `№`, `+` …), kichik harf, bo‘shliqlar.

Kelishilgan qarorlar: 100 → "bir yuz", 1000 → "bir ming" (rasmiy o‘qish); akronimlar o‘zbekcha harf nomlari bilan.

**Phonemizer**: `sh ch ng gʻ oʻ` digraflar, `sʼh` → s+h, tutuq → ʔ, `x` → χ, `j` → dʒ (ʒ faqat lug‘atdagi
o‘zlashma so‘zlarda), `a` → ɑ q/gʻ/x yonida, aks holda æ; `v` → w unlidan keyin undosh oldidan;
so‘z oxiridagi b/d/g jarangsizlanadi. Urg‘u: oxirgi bo‘g‘in + `lexicon.STRESS` istisnolari.

Hozircha **yo‘q** (ataylab): qo‘shimchaga bog‘liq urg‘u (`-mi`, `-man`, `-chi` urg‘usiz) — morfologik tahlil kerak;
`i` reduksiyasi (ɨ) qoidasi; gap darajasidagi intonatsiya belgilari. Lug‘at va urg‘u istisnolarini ona tili
egasi ko‘rib chiqishi kerak. Inventar teacher checkpoint'iga moslashtirilishi mumkin (quyida).

## Ochiq qadam: teacher checkpoint audit

Teacher sifatida `Inomjonov/tiny-voca-uzbek-navoiy` (4.53M, 7.695 soat synthetic male) ishlatiladi.
Uning state_dict'i hali tekshirilmagan (bu sessiyada huggingface.co ga ruxsat yo‘q edi). Checkpoint yuklab olingach:

```bash
PYTHONPATH=src python -m tiny_voca_uz.teacher_audit path/to/checkpoint.pt > docs/teacher_audit.md
```

Bu teacher'ning vocab hajmi, speaker soni, stress darajalari va qatlamlarini o‘qiydi. Agar uning fonema inventari
`symbols.py` dan farq qilsa, student teacher'nikini qabul qiladi (distillation uchun token id'lar mos bo‘lishi shart).

## Keyingi bosqichlar

1. Teacher audit (yuqorida) va inventarni moslash.
2. Frontend davomi: morfologik urg‘u qoidalari, kengroq lug‘at, ona tili egasi bilan tekshiruv, sentence splitting.
3. Teacher inference cache (duration, pitch/voicing/energy, hidden'lar, waveform).
4. Distillation training (spectral + duration + prosody + feature loss), keyin adversarial refinement.
5. Evaluation (ASR WER/CER, MOS, CPU RTF) va ONNX/INT8 export.

## Litsenziya

Apache-2.0. Original tiny-voca kodidan olingan qismlar `NOTICE` faylida ko‘rsatilgan.
