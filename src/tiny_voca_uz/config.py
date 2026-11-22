"""Model configuration. Defaults are the 1M student (Tiny-Voca-UZ-1M);
``teacher_config()`` reproduces the original 4.5M architecture for audits."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from .symbols import STRESS_LEVELS, SYMBOLS


@dataclass(frozen=True)
class Config:
    # Acoustic features: identical to the teacher so its cached targets are reusable.
    sample_rate: int = 24000
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f0_min: float = 50.0
    f0_max: float = 600.0
    speakers: tuple[str, ...] = ("M1",)
    vocab_size: int = len(SYMBOLS)
    stress_levels: int = STRESS_LEVELS
    # Transformers. 96 = half the teacher width, so feature distillation needs
    # only a 96 -> 192 linear projection (training-only, not counted).
    channels: int = 96
    heads: int = 4
    feedforward: int = 384
    kernel: int = 7
    encoder_layers: int = 3
    prosody_layers: int = 2
    acoustic_layers: int = 3
    dropout: float = 0.1
    predictor_channels: int = 96
    # Source-filter waveform head (unchanged from the teacher).
    synth_fft: int = 512
    synth_hop: int = 128
    noise_fft: int = 256
    noise_hop: int = 64
    cep_causal: int = 80
    cep_anti: int = 16
    noise_cep: int = 32
    harmonic_limit: float = 0.47
    max_tokens: int = 512
    max_frames: int = 3000
    mel_mean: float = -5.0
    mel_scale: float = 5.0
    energy_mean: float = -4.0
    energy_scale: float = 2.0
    pitch_reference_hz: float = 200.0

    def __post_init__(self):
        if self.vocab_size < 2 or self.stress_levels < 1:
            raise ValueError("vocab_size and stress_levels must be positive")
        if self.channels % self.heads or (self.channels // self.heads) % 2:
            raise ValueError("channels must be divisible by heads with an even head width")
        if not self.speakers or len(set(self.speakers)) != len(self.speakers):
            raise ValueError("speaker names must be unique and nonempty")
        for fft, hop in ((self.synth_fft, self.synth_hop), (self.noise_fft, self.noise_hop)):
            if fft != 4 * hop or self.hop % hop:
                raise ValueError("synthesis windows need 4x overlap and a hop dividing the acoustic hop")
        if self.cep_causal + self.cep_anti >= self.synth_fft // 2 or self.noise_cep >= self.noise_fft // 2:
            raise ValueError("cepstrum orders must be well below half the synthesis FFT size")
        if min(self.encoder_layers, self.prosody_layers, self.acoustic_layers, self.kernel) < 1 or not self.kernel % 2:
            raise ValueError("layer counts must be positive and the kernel odd")
        if min(self.mel_scale, self.energy_scale) <= 0 or not 0 < self.harmonic_limit < 0.5:
            raise ValueError("invalid normalization scale or harmonic limit")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "speakers": tuple(value["speakers"])})

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def teacher_config(**overrides):
    """The original Tiny-Voca v2 architecture (4.5M). With the English inventory
    (76 symbols, 3 stress levels, 2 voices) it counts exactly 4,537,291."""
    return replace(Config(), channels=192, feedforward=640, encoder_layers=4, prosody_layers=2, acoustic_layers=5,
                   predictor_channels=128, **overrides)


def english_teacher_config():
    return teacher_config(vocab_size=76, stress_levels=3, speakers=("F2", "M2"))
