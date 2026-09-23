"""APSR: frequency-resolved enrollment memory with shared R4 refinement.

Read forward() for the full computation; separator.py and memory.py contain
the refinement schedule and the time recurrence respectively.
"""

from __future__ import annotations

import torch
from torch import nn

from .frontend import ERB, SFE
from .encoder import SpeechEncoder
from .enrollment import GatedAttentiveEnrollmentPooling
from .blocks import SeparatorBlock
from .memory import TargetAnchoredTimeConstantMemory
from .decoder import SpeechDecoder, ComplexMaskHead
from .separator import ProgressiveRefinement


class APSR(ProgressiveRefinement, nn.Module):
    """The paper's P model, 8 kHz, 64 channels, eight tokens, four rounds.

    Input: mixture (B,L), enrollment (B,E) or (1,E), float32 waveforms.
    Output: target (B,1,L). All enrollment audio is available in advance.
    The separator is causal in mixture frames; the centered STFT has 16 ms
    analysis lookahead. This API processes complete utterances.
    """

    # Historical tensor names are preserved for the already released E102.
    SCHEMA_KEYS = (
        "_m4_gtless_m3_schema_version",
        "_m5_apsr_r4_schema_version",
        "_m5_ablation_schema_version",
        "_m5_ablation_v3_schema_version",
        "_m5_ablation_v6_schema_version",
    )

    def __init__(self):
        super().__init__()
        self.n_fft = self.win_len = 256
        self.hop_len = 64
        self.sample_rate = 8000
        self.num_sources = 1
        self.stft_center = True
        self.normalize_enrollment = True
        self.hidden_channels = 64
        self.enroll_pool_tokens = 8
        self.refinement_rounds = 4
        self.trust_eta = self.trust_kappa = 0.5
        self.erb = ERB(33, 32, nfft=256, high_lim=4000, fs=8000)
        self.sfe = SFE(3, 1)
        self.encoder = SpeechEncoder(64)
        self.enr_pool = GatedAttentiveEnrollmentPooling(64, num_tokens=8, att_dim=32)
        self.sep_token = nn.Parameter(torch.zeros(1, 64, 1, 65))
        self.dp_blocks = nn.ModuleList([SeparatorBlock(64, 65, 64) for _ in range(4)])
        for block in self.dp_blocks:
            block.direct_prefix_tokens = 9
            block.car_attn.direct_prefix_tokens = 9
            block.target_memory = TargetAnchoredTimeConstantMemory(
                channels=64,
                memory_dim=16,
                num_enroll_tokens=8,
                hop_seconds=0.008,
                tau_min_seconds=0.25,
                tau_max_seconds=4.0,
                tau_init_seconds=2.0,
                write_bias=-2.0,
                beta_init=0.1,
                beta_max=0.5,
                zero_init_output=True,
                scan_chunk_frames=64,
            )
        self.decoder = SpeechDecoder(64)
        self.mask = ComplexMaskHead(64, num_sources=1, nfft=256, noise_sink=True)
        for key in self.SCHEMA_KEYS:
            self.register_buffer(key, torch.tensor(1, dtype=torch.int64))
        self._last_aux = None

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        """Encode both inputs separately; condition through memory; decode target."""
        if mixture.ndim != 2 or enrollment.ndim != 2:
            raise ValueError("Expected mixture (B,L) and enrollment (B,E)")
        if min(mixture.shape[-1], enrollment.shape[-1]) <= self.n_fft // 2:
            raise ValueError(
                "Both inputs must exceed 128 samples for reflect-padded STFT"
            )
        features, spectrum = self._extract_features(mixture, return_spec=True)
        features, skips = self.encoder(features)
        enrollment_features = self._encode_enrollment_features(enrollment)
        enrollment_features = self._expand_enrollment_batch(
            enrollment_features, mixture.shape[0]
        )
        tokens = self.enr_pool(
            enrollment_features.to(device=features.device, dtype=features.dtype)
        )
        features, tokens = self._conditioned_blocks(features, tokens)
        decoded = self.erb.bs(self.decoder(features, skips))
        target_spectrum, aux = self.mask(decoded, spectrum)
        aux["enr_pooled"] = tokens
        aux["target_spectrum_ri"] = aux["grouped_specs"][:, 0]
        self._last_aux = aux
        return self._specs_to_waveform(
            target_spectrum, self._stft_kwargs(mixture), mixture.shape[-1]
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("APSR requires strict checkpoint loading")
        state = {k.removeprefix("module."): v for k, v in state_dict.items()}
        if len(state) != len(state_dict):
            raise ValueError("Checkpoint contains colliding parameter names")
        for key in self.SCHEMA_KEYS:
            value = state.get(key)
            if (
                not torch.is_tensor(value)
                or value.shape != ()
                or value.dtype != torch.int64
                or value.item() != 1
            ):
                raise ValueError(f"Incompatible checkpoint schema: {key}")
        return super().load_state_dict(state, strict=True, assign=assign)

    @classmethod
    def from_pretrained(cls, checkpoint, device="cpu"):
        """Load the released tensor-only state dict or a public trainer checkpoint."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"] if "model" in payload else payload
        model = cls()
        model.load_state_dict(state)
        model._checkpoint_info = {
            "epoch": payload.get("epoch", "UNKNOWN"),
            "best_epoch": payload.get("best_epoch", "UNKNOWN"),
            "training_scope": "SMOKE_ONLY"
            if payload.get("smoke_steps") is not None
            else "UNKNOWN",
        }
        return model.to(device).eval()

    def _stft_kwargs(self, x):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": torch.hann_window(self.win_len, device=x.device, dtype=x.dtype),
            "onesided": True,
            "center": self.stft_center,
        }

    def _extract_features(self, wav, return_spec=False):
        spec = torch.stft(wav, **self._stft_kwargs(wav), return_complex=True)
        spec = torch.view_as_real(spec)
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        feat = self.erb.bm(feat)
        feat = self.sfe(feat)
        if not return_spec:
            return feat
        spec_ri = spec.permute(0, 3, 2, 1)
        return (feat, spec_ri)

    def _maybe_normalize_enrollment(self, enrollment):
        if not self.normalize_enrollment:
            return enrollment
        return enrollment / (enrollment.std(dim=-1, keepdim=True) + 1e-08)

    def _encode_enrollment_features(self, enrollment):
        enrollment = self._maybe_normalize_enrollment(enrollment)
        enr_feat = self._extract_features(enrollment, return_spec=False)
        enr_enc, _ = self.encoder(enr_feat)
        return enr_enc

    def _expand_enrollment_batch(self, enr_pooled, batch_size):
        if enr_pooled.shape[0] == batch_size:
            return enr_pooled
        if enr_pooled.shape[0] == 1:
            return enr_pooled.expand(batch_size, -1, -1, -1)
        raise ValueError(
            f"Enrollment batch {enr_pooled.shape[0]} does not match mixture batch {batch_size}"
        )

    def _specs_to_waveform(self, specs_enh, stft_kwargs, length):
        outputs = []
        for src_idx in range(self.num_sources):
            spec_k = specs_enh[:, src_idx]
            spec_k = spec_k.permute(0, 3, 2, 1)
            spec_k = torch.complex(spec_k[..., 0], spec_k[..., 1])
            outputs.append(torch.istft(spec_k, length=length, **stft_kwargs))
        return torch.stack(outputs, dim=1)

    def training_stft_ri(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2 or waveform.shape[-1] <= 0:
            raise ValueError("waveform must have shape (B,L) or (B,1,L)")
        spectrum = torch.stft(
            waveform, **self._stft_kwargs(waveform), return_complex=True
        )
        return (
            torch.stack([spectrum.real, spectrum.imag], dim=1)
            .permute(0, 1, 3, 2)
            .contiguous()
        )


# Compatibility alias for integrations using the former public class name.
APSRP = APSR
