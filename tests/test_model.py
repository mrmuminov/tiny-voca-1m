import time
import unittest
import torch

from tiny_voca_uz.config import Config, english_teacher_config, teacher_config
from tiny_voca_uz.model import TinyVocaUZ
from tiny_voca_uz.params import parameter_report
from tiny_voca_uz.symbols import SYMBOL_TO_ID
from tiny_voca_uz.teacher_audit import audit

# "Bugun havo yaxshi." as a hand-written token sequence (front end not built yet).
SYMBOLS = ("<bos>", "b", "u", "ɡ", "u", "n", "<sp>", "h", "ɑ", "v", "ɒ", "<sp>", "j", "ɑ", "χ", "ʃ", "i", ".", "<eos>")
STRESS = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0)
DURATIONS = (3, 4, 5, 4, 9, 5, 0, 5, 8, 4, 10, 2, 4, 7, 6, 8, 12, 6, 5)


def batch(rows=1):
    tokens = torch.tensor([[SYMBOL_TO_ID[s] for s in SYMBOLS]] * rows)
    stress = torch.tensor([STRESS] * rows)
    lengths = torch.full((rows,), len(SYMBOLS))
    return tokens, stress, lengths, torch.zeros(rows, dtype=torch.long)


class ParameterTests(unittest.TestCase):
    def test_original_architecture_count_is_reproduced(self):
        self.assertEqual(parameter_report(english_teacher_config())["total"], 4_537_291)

    def test_student_is_about_one_million(self):
        report = parameter_report(Config())
        self.assertGreaterEqual(report["total"], 950_000)
        self.assertLessEqual(report["total"], 1_050_000)
        self.assertLess(report["total"] * 4.5, parameter_report(teacher_config())["total"])

    def test_teacher_audit_reads_the_original_layout(self):
        state = TinyVocaUZ(english_teacher_config()).state_dict()
        result = audit(state)
        self.assertEqual(result["inferred"]["parameters"], 4_537_291)
        self.assertEqual((result["inferred"]["vocab_size"], result["inferred"]["speakers"]), (76, 2))
        self.assertEqual(result["inferred"]["acoustic_layers"], 5)
        self.assertFalse(result["mismatched"] or result["missing"] or result["unexpected"])


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = TinyVocaUZ(Config()).eval()

    def test_teacher_forced_forward(self):
        tokens, stress, lengths, speakers = batch(2)
        model, c = self.model, self.model.config
        with torch.no_grad():
            hidden, mask = model.encode(tokens, stress, lengths, speakers)
            self.assertEqual(hidden.shape, (2, len(SYMBOLS), c.channels))
            durations = torch.tensor([DURATIONS] * 2)
            frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, speakers)
            self.assertEqual(int(frame_lengths[0]), sum(DURATIONS))
            prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
            self.assertEqual(prosody.shape, (2, sum(DURATIONS), 3))
            acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
            wave = model.vocoder(acoustic, model.frequency(prosody[..., 0]), frame_lengths)
        self.assertEqual(wave.shape, (2, sum(DURATIONS) * c.hop))
        self.assertTrue(torch.isfinite(wave).all())

    def test_free_running_synthesis(self):
        tokens, stress, lengths, speakers = batch()
        start = time.perf_counter()
        wave, samples, extras = self.model.synthesize(tokens, stress, lengths, speakers, return_hidden=True)
        elapsed = time.perf_counter() - start
        self.assertEqual(int(samples[0]), wave.shape[1])
        self.assertEqual(wave.shape[1], int(extras["durations"].sum()) * self.model.config.hop)
        self.assertTrue(torch.isfinite(wave).all())
        self.assertEqual(extras["encoder"].shape[-1], 96)
        self.assertEqual(extras["acoustic"].shape[:2], extras["prosody"].shape[:2])
        seconds = wave.shape[1] / self.model.config.sample_rate
        print(f"\n[untrained, CPU] {seconds:.2f}s audio in {elapsed * 1000:.0f} ms (RTF {elapsed / seconds:.3f})")

    def test_padding_does_not_change_audio(self):
        tokens, stress, lengths, speakers = batch(2)
        short = 12
        lengths = torch.tensor([short, len(SYMBOLS)])
        tokens[0, short:], stress[0, short:] = 0, 0
        together, n, _ = self.model.synthesize(tokens, stress, lengths, speakers, seed=3)
        alone, m, _ = self.model.synthesize(tokens[:1, :short], stress[:1, :short], lengths[:1], speakers[:1], seed=3)
        self.assertEqual(int(n[0]), int(m[0]))
        torch.testing.assert_close(together[0, :int(n[0])], alone[0], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
