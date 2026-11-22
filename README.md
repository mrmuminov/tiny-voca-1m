# Tiny-Voca-UZ-1M

Taxminan 1 million parametrli, bitta tilga (o‘zbek tili) va bitta ovozga ixtisoslashgan TTS model.
24 kHz, CPU inference, keyinchalik ONNX. Arxitektura [tiny-voca](https://github.com/n4ze3m/tiny-voca) (Apache-2.0)
asosida: FastSpeech 2 uslubidagi transformer + source-filter synthesizer (neural vocoder yo‘q).

> Holat: **1-deliverable** — audit, parametr hisobi, 1M student arxitekturasi va forward testlar.
> Frontend, dataset, distillation va training hali yo‘q.

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
2. O‘zbek frontend: normalizer (raqam, sana, foiz, vaqt), phonemizer (o‘, g‘, sh, ch, ng, tutuq belgisi), urg‘u.
3. Teacher inference cache (duration, pitch/voicing/energy, hidden'lar, waveform).
4. Distillation training (spectral + duration + prosody + feature loss), keyin adversarial refinement.
5. Evaluation (ASR WER/CER, MOS, CPU RTF) va ONNX/INT8 export.

## Litsenziya

Apache-2.0. Original tiny-voca kodidan olingan qismlar `NOTICE` faylida ko‘rsatilgan.
