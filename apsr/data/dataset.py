"""
WSJ0-2MIX/WHAM-style dataset loader for target speaker extraction.

Expected split directory:
    /path/to/wav8k/min/tr/
    ├── mix_clean/
    ├── mix_both/      # optional noisy mixture
    ├── noise/         # optional
    ├── s1/
    └── s2/

Returns triplets:
    mix:        (L,)
    target:     (L,)
    enrollment: (L_enroll,)

Enrollment is selected from another utterance of the same WSJ0 speaker when
possible. If the speaker has no other available utterance in the split, the
loader falls back to a non-overlapping crop from the same source file.
"""

from __future__ import annotations
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def _parse_wsj0_mix_filename(fname: str) -> Tuple[str, str, str, str]:
    """Parse WHAM/WSJ0-2MIX filename into source ids and speaker ids.

    Example:
        024o030i_0.89792_01la0107_-0.89792.wav

    WSJ0 speaker ids are encoded by the first three characters of each source
    utterance id.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected WSJ0-2MIX filename: {fname}")
    src1_id = parts[0]
    src2_id = parts[2]
    return (src1_id, src1_id[:3], src2_id, src2_id[:3])


def _parse_libri2mix_filename(fname: str) -> Tuple[str, str, str, str]:
    """Parse Libri2Mix filename into source ids and speaker ids.

    Example:
        103-1240-0003_1235-135887-0017.wav

    Libri speaker ids are encoded before the first dash of each source id.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected Libri2Mix filename: {fname}")
    src1_id = parts[0]
    src2_id = parts[1]
    return (src1_id, src1_id.split("-")[0], src2_id, src2_id.split("-")[0])


class WSJ02MixTSEDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        sample_rate: int = 8000,
        segment: Optional[float] = 3.0,
        enroll_segment: Optional[float] = 3.0,
        normalize: bool = False,
        mix_type: str = "mix_clean",
        random_start: bool = True,
        deterministic_eval: bool = False,
        target_mode: str = "random",
        drop_short: bool = False,
        manifest_dir: Optional[str] = None,
        dataset_type: str = "tree",
        enrollment_policy: str = "auto",
        scp_group_by_mixture: bool = False,
        scp_pair_targets: bool = False,
        filename_style: str = "auto",
        waveform_scale: float = 1.0,
        enrollment_loudness_norm: bool = False,
        enrollment_loudness_metadata_dir: Optional[str] = None,
        enrollment_loudness_target_rms: float = 0.073,
        enrollment_loudness_peak_limit: float = 0.9,
        enrollment_loudness_missing: str = "raise",
        dynamic_source_rebalance: bool = False,
        dynamic_sir_min_db: float = -3.0,
        dynamic_sir_max_db: float = 3.0,
        dynamic_rebalance_probability: float = 1.0,
        dynamic_rebalance_peak_limit: float = 0.9,
    ):
        super().__init__()
        self.dataset_type = str(dataset_type)
        self.manifest_dir = (
            None if manifest_dir is None else os.path.abspath(manifest_dir)
        )
        self.is_scp_mode = (
            self.manifest_dir is not None
            or self.dataset_type.lower() in {"scp", "wsj0_2mix_tse_scp"}
        )
        if data_dir is None and (not self.is_scp_mode):
            raise ValueError("data_dir is required for tree-mode WSJ0-2MIX loading.")
        self.data_dir = None if data_dir is None else os.path.abspath(data_dir)
        self.sample_rate = int(sample_rate)
        self.segment = segment
        self.seg_len = (
            None if segment is None else int(round(float(segment) * self.sample_rate))
        )
        self.enroll_segment = enroll_segment
        self.enroll_len = (
            None
            if enroll_segment is None
            else int(round(float(enroll_segment) * self.sample_rate))
        )
        self.normalize = bool(normalize)
        self.mix_type = str(mix_type)
        self.random_start = bool(random_start)
        self.deterministic_eval = bool(deterministic_eval)
        self.target_mode = str(target_mode)
        self.drop_short = bool(drop_short)
        self.enrollment_policy = str(enrollment_policy).lower()
        self.scp_group_by_mixture = bool(scp_group_by_mixture)
        self.scp_pair_targets = bool(scp_pair_targets)
        self.filename_style = str(filename_style).lower()
        self.waveform_scale = float(waveform_scale)
        self.enrollment_loudness_norm = bool(enrollment_loudness_norm)
        self.enrollment_loudness_metadata_dir = (
            None
            if enrollment_loudness_metadata_dir is None
            else os.path.abspath(enrollment_loudness_metadata_dir)
        )
        self.enrollment_loudness_target_rms = float(enrollment_loudness_target_rms)
        self.enrollment_loudness_peak_limit = float(enrollment_loudness_peak_limit)
        self.enrollment_loudness_missing = str(enrollment_loudness_missing).lower()
        self.dynamic_source_rebalance = bool(dynamic_source_rebalance)
        self.dynamic_sir_min_db = float(dynamic_sir_min_db)
        self.dynamic_sir_max_db = float(dynamic_sir_max_db)
        self.dynamic_rebalance_probability = float(dynamic_rebalance_probability)
        self.dynamic_rebalance_peak_limit = float(dynamic_rebalance_peak_limit)
        self.eps = 1e-08
        if self.target_mode not in ("random", "s1", "s2", "alternate", "both"):
            raise ValueError(
                "target_mode must be one of: random, s1, s2, alternate, both"
            )
        if self.filename_style not in (
            "auto",
            "wsj0_2mix",
            "wsj0",
            "libri2mix",
            "libri",
        ):
            raise ValueError(
                "filename_style must be one of: auto, wsj0_2mix, wsj0, libri2mix, libri"
            )
        if self.enrollment_policy == "auto":
            self.enrollment_policy = (
                "fixed" if self.deterministic_eval else "directory_random"
            )
        if self.enrollment_policy not in ("fixed", "directory_random", "source_random"):
            raise ValueError(
                "enrollment_policy must be one of: auto, fixed, directory_random, source_random"
            )
        if self.enrollment_loudness_missing not in ("raise", "warn", "ignore"):
            raise ValueError(
                "enrollment_loudness_missing must be one of: raise, warn, ignore"
            )
        if self.dynamic_source_rebalance and self.deterministic_eval:
            raise ValueError(
                "dynamic_source_rebalance is training-only and must be false when deterministic_eval=true"
            )
        if not np.isfinite(self.dynamic_sir_min_db) or not np.isfinite(
            self.dynamic_sir_max_db
        ):
            raise ValueError("dynamic SIR limits must be finite")
        if self.dynamic_sir_min_db > self.dynamic_sir_max_db:
            raise ValueError("dynamic_sir_min_db must not exceed dynamic_sir_max_db")
        if not 0.0 <= self.dynamic_rebalance_probability <= 1.0:
            raise ValueError("dynamic_rebalance_probability must be in [0,1]")
        if not 0.0 < self.dynamic_rebalance_peak_limit <= 1.0:
            raise ValueError("dynamic_rebalance_peak_limit must be in (0,1]")
        self._aux_dir_cache: Dict[str, List[str]] = {}
        self._warned_missing_loudness: set[str] = set()
        self.enrollment_activlev = (
            self._load_activlev_metadata() if self.enrollment_loudness_norm else None
        )
        self.entry_groups: List[List[int]] = []
        self.scp_spk_to_refs = defaultdict(list)
        if self.is_scp_mode:
            self._init_scp_mode()
            return
        self.mix_dir = os.path.join(self.data_dir, self.mix_type)
        if not os.path.isdir(self.mix_dir):
            raise FileNotFoundError(
                f"Mix directory not found: {self.mix_dir}\nAvailable: {(os.listdir(self.data_dir) if os.path.isdir(self.data_dir) else 'N/A')}"
            )
        for src in ("s1", "s2"):
            src_dir = os.path.join(self.data_dir, src)
            if not os.path.isdir(src_dir):
                raise FileNotFoundError(f"Source directory not found: {src_dir}")
        candidates = sorted((f for f in os.listdir(self.mix_dir) if f.endswith(".wav")))
        self.files: List[str] = []
        self.lengths: List[int] = []
        for fname in candidates:
            paths = [
                os.path.join(self.mix_dir, fname),
                os.path.join(self.data_dir, "s1", fname),
                os.path.join(self.data_dir, "s2", fname),
            ]
            if not all((os.path.exists(path) for path in paths)):
                continue
            info = sf.info(paths[0])
            if info.samplerate != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz, got {info.samplerate} Hz for {paths[0]}"
                )
            if (
                self.drop_short
                and self.seg_len is not None
                and (info.frames < self.seg_len)
            ):
                continue
            self.files.append(fname)
            self.lengths.append(int(info.frames))
        if not self.files:
            raise RuntimeError(f"No usable WSJ0-2MIX wavs found in {self.mix_dir}")
        self.spk_to_refs = defaultdict(list)
        for fname in self.files:
            _src1_id, spk1, _src2_id, spk2 = self._parse_mix_filename(fname)
            self.spk_to_refs[spk1].append((fname, "s1"))
            self.spk_to_refs[spk2].append((fname, "s2"))
        print(
            f"[WSJ02MixTSE] {len(self.files)} mixtures from {self.data_dir}/{self.mix_type}"
        )
        print(f"[WSJ02MixTSE] Found {len(self.spk_to_refs)} speaker ids.")
        if self.seg_len is not None:
            mode = (
                "deterministic"
                if self.deterministic_eval
                else "random"
                if self.random_start
                else "first"
            )
            print(
                f"[WSJ02MixTSE] Segment: {self.segment:.2f}s = {self.seg_len} samples ({mode} crop)"
            )
        if self.dynamic_source_rebalance:
            print(
                f"[WSJ02MixTSE] Dynamic source rebalance: SIR=[{self.dynamic_sir_min_db:.1f},{self.dynamic_sir_max_db:.1f}] dB, p={self.dynamic_rebalance_probability:.2f}"
            )

    @staticmethod
    def _read_scp(path: str) -> Tuple[List[str], Dict[str, str]]:
        keys: List[str] = []
        mapping: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Bad scp line {path}:{line_no}: {line!r}")
                key, wav_path = parts
                if key in mapping:
                    raise ValueError(f"Duplicate scp key {key!r} in {path}")
                keys.append(key)
                mapping[key] = wav_path
        return (keys, mapping)

    def _load_activlev_metadata(self) -> Dict[str, float]:
        if self.enrollment_loudness_metadata_dir is None:
            raise ValueError(
                "enrollment_loudness_metadata_dir is required when enrollment_loudness_norm=true"
            )
        meta_path = Path(self.enrollment_loudness_metadata_dir)
        if meta_path.is_dir():
            files = sorted(meta_path.glob("*.txt"))
        else:
            files = [meta_path]
        if not files:
            raise FileNotFoundError(
                f"No active-level metadata files found under {meta_path}"
            )
        activlev: Dict[str, float] = {}
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 2:
                        raise ValueError(
                            f"Bad active-level line {path}:{line_no}: {line!r}"
                        )
                    utt_id, value = parts
                    activlev[utt_id] = float(value)
        if not activlev:
            raise RuntimeError(f"No active-level entries loaded from {meta_path}")
        return activlev

    def _apply_enrollment_loudness_norm(
        self, audio: np.ndarray, path: str
    ) -> np.ndarray:
        if not self.enrollment_loudness_norm:
            return audio
        assert self.enrollment_activlev is not None
        utt_id = Path(path).stem
        active_level = self.enrollment_activlev.get(utt_id)
        if active_level is None:
            if self.enrollment_loudness_missing == "raise":
                raise KeyError(
                    f"Missing active-level metadata for enrollment utterance {utt_id!r}"
                )
            if (
                self.enrollment_loudness_missing == "warn"
                and utt_id not in self._warned_missing_loudness
            ):
                print(
                    f"[WSJ02MixTSE] WARNING: missing active-level metadata for {utt_id}; leaving enrollment unscaled"
                )
                self._warned_missing_loudness.add(utt_id)
            return audio
        scale = self.enrollment_loudness_target_rms / (
            float(active_level) ** 0.5 + self.eps
        )
        audio = (audio.astype(np.float32, copy=False) * np.float32(scale)).astype(
            np.float32, copy=False
        )
        if self.enrollment_loudness_peak_limit > 0:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > self.enrollment_loudness_peak_limit:
                audio = audio / np.float32(peak / self.enrollment_loudness_peak_limit)
        return audio.astype(np.float32, copy=False)

    def _init_scp_mode(self) -> None:
        if self.manifest_dir is None:
            raise ValueError("manifest_dir is required for scp-mode WSJ0-2MIX loading.")
        manifest = Path(self.manifest_dir)
        mix_keys, mix_map = self._read_scp(str(manifest / "mix.scp"))
        ref_keys, ref_map = self._read_scp(str(manifest / "ref.scp"))
        aux_keys, aux_map = self._read_scp(str(manifest / "aux.scp"))
        if mix_keys != ref_keys or mix_keys != aux_keys:
            raise ValueError(
                f"mix/ref/aux scp key order mismatch under {self.manifest_dir}. Regenerate manifests before training."
            )
        self.entries = []
        self.lengths = []
        for key in mix_keys:
            mix_path = os.path.abspath(mix_map[key])
            ref_path = os.path.abspath(ref_map[key])
            aux_path = os.path.abspath(aux_map[key])
            for path in (mix_path, ref_path, aux_path):
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
            mix_info = sf.info(mix_path)
            if mix_info.samplerate != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz, got {mix_info.samplerate} Hz for {mix_path}"
                )
            if (
                self.drop_short
                and self.seg_len is not None
                and (mix_info.frames < self.seg_len)
            ):
                continue
            self.entries.append((key, mix_path, ref_path, aux_path))
            self.lengths.append(int(mix_info.frames))
        if not self.entries:
            raise RuntimeError(f"No usable scp entries found in {self.manifest_dir}")
        self.scp_spk_to_refs = defaultdict(list)
        for entry_idx, (_key, mix_path, ref_path, _aux_path) in enumerate(self.entries):
            target_src = self._target_src_from_ref_path(ref_path)
            src1_id, spk1, src2_id, spk2 = self._parse_mix_filename(
                os.path.basename(mix_path)
            )
            target_spk = spk1 if target_src == "s1" else spk2
            target_src_id = src1_id if target_src == "s1" else src2_id
            self.scp_spk_to_refs[target_spk].append((entry_idx, target_src_id))
        if self.scp_group_by_mixture or self.scp_pair_targets:
            groups = defaultdict(list)
            for entry_idx, (_key, mix_path, _ref_path, _aux_path) in enumerate(
                self.entries
            ):
                groups[mix_path].append(entry_idx)
            self.entry_groups = [groups[mix_path] for mix_path in sorted(groups)]
            for group in self.entry_groups:
                group.sort(
                    key=lambda entry_idx: self._target_src_from_ref_path(
                        self.entries[entry_idx][2]
                    )
                )
            print(
                f"[WSJ02MixTSE:SCP] {len(self.entries)} target-speaker cases grouped into {len(self.entry_groups)} mixtures from {self.manifest_dir}"
            )
            if self.scp_pair_targets:
                bad_groups = sum((1 for group in self.entry_groups if len(group) != 2))
                if bad_groups:
                    raise ValueError(
                        f"scp_pair_targets requires exactly two target entries per mixture; found {bad_groups} bad groups under {self.manifest_dir}"
                    )
        else:
            print(
                f"[WSJ02MixTSE:SCP] {len(self.entries)} target-speaker cases from {self.manifest_dir}"
            )
        print(f"[WSJ02MixTSE:SCP] Enrollment policy: {self.enrollment_policy}")
        if self.seg_len is not None:
            mode = (
                "deterministic"
                if self.deterministic_eval
                else "random"
                if self.random_start
                else "first"
            )
            print(
                f"[WSJ02MixTSE:SCP] Segment: {self.segment:.2f}s = {self.seg_len} samples ({mode} crop)"
            )
        else:
            print("[WSJ02MixTSE:SCP] Segment: full utterance")
        if self.enroll_len is None:
            print("[WSJ02MixTSE:SCP] Enrollment: full utterance")
        else:
            print(
                f"[WSJ02MixTSE:SCP] Enrollment crop: {float(self.enroll_segment):.2f}s = {self.enroll_len} samples"
            )
        if self.dynamic_source_rebalance:
            print(
                f"[WSJ02MixTSE:SCP] Dynamic source rebalance: SIR=[{self.dynamic_sir_min_db:.1f},{self.dynamic_sir_max_db:.1f}] dB, p={self.dynamic_rebalance_probability:.2f}"
            )

    def __len__(self) -> int:
        if self.is_scp_mode:
            if self.scp_pair_targets:
                return len(self.entry_groups)
            if self.scp_group_by_mixture:
                return len(self.entry_groups)
            return len(self.entries)
        if self.target_mode == "both":
            return 2 * len(self.files)
        return len(self.files)

    @staticmethod
    def _stable_int(*parts) -> int:
        payload = "|".join((str(p) for p in parts)).encode("utf-8")
        return int.from_bytes(
            hashlib.md5(payload).digest()[:8], byteorder="big", signed=False
        )

    def _sample_offset(self, low: int, high: int, *parts) -> int:
        if high <= low:
            return low
        if self.deterministic_eval:
            return low + self._stable_int(*parts) % (high - low)
        return int(np.random.randint(low, high))

    def _dynamic_rebalance_sources(
        self, first: torch.Tensor, second: torch.Tensor, reference_mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create one same-mixture two-target trial at a sampled source SIR.

        Each source is first RMS-normalized, opposite half-SIR gains are
        applied, and one common gain restores the original mixture RMS.  A
        common peak limiter preserves exact additivity: returned
        ``first + second == mixture`` up to floating-point roundoff.

        The operation is deliberately waveform-only and training-only.  It
        changes neither evaluation examples nor inference computation.
        """
        if not self.dynamic_source_rebalance:
            return (reference_mix, first, second)
        if np.random.random() >= self.dynamic_rebalance_probability:
            return (reference_mix, first, second)
        if first.shape != second.shape or first.shape != reference_mix.shape:
            raise ValueError("dynamic source rebalance requires aligned waveforms")
        first32 = first.float()
        second32 = second.float()
        mix32 = reference_mix.float()
        first_rms = first32.square().mean().sqrt()
        second_rms = second32.square().mean().sqrt()
        mix_rms = mix32.square().mean().sqrt()
        if (
            float(first_rms) <= self.eps
            or float(second_rms) <= self.eps
            or float(mix_rms) <= self.eps
        ):
            return (reference_mix, first, second)
        sir_db = float(
            np.random.uniform(self.dynamic_sir_min_db, self.dynamic_sir_max_db)
        )
        half_ratio = 10.0 ** (sir_db / 40.0)
        first_scaled = first32 * (half_ratio / (first_rms + self.eps))
        second_scaled = second32 * (1.0 / half_ratio / (second_rms + self.eps))
        remixed = first_scaled + second_scaled
        common_gain = mix_rms / (remixed.square().mean().sqrt() + self.eps)
        first_scaled = first_scaled * common_gain
        second_scaled = second_scaled * common_gain
        remixed = first_scaled + second_scaled
        peak = remixed.abs().amax()
        if float(peak) > self.dynamic_rebalance_peak_limit:
            peak_gain = self.dynamic_rebalance_peak_limit / (peak + self.eps)
            first_scaled = first_scaled * peak_gain
            second_scaled = second_scaled * peak_gain
            remixed = first_scaled + second_scaled
        return (
            remixed.to(dtype=reference_mix.dtype),
            first_scaled.to(dtype=first.dtype),
            second_scaled.to(dtype=second.dtype),
        )

    def _dynamic_rebalance_target_residual(
        self, mix: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = mix - target
        remixed, target_scaled, _residual_scaled = self._dynamic_rebalance_sources(
            target, residual, mix
        )
        return (remixed, target_scaled)

    def _target_index(self, idx: int, fname: str) -> int:
        if self.target_mode == "s1":
            return 0
        if self.target_mode == "s2":
            return 1
        if self.target_mode == "both":
            return idx % 2
        if self.target_mode == "alternate":
            return idx % 2
        if self.deterministic_eval:
            return self._stable_int(fname, "target") % 2
        return int(np.random.randint(0, 2))

    def _parse_mix_filename(self, fname: str) -> Tuple[str, str, str, str]:
        if self.filename_style in ("wsj0_2mix", "wsj0"):
            return _parse_wsj0_mix_filename(fname)
        if self.filename_style in ("libri2mix", "libri"):
            return _parse_libri2mix_filename(fname)
        stem = os.path.splitext(os.path.basename(fname))[0]
        parts = stem.split("_")
        if len(parts) == 2:
            return _parse_libri2mix_filename(fname)
        return _parse_wsj0_mix_filename(fname)

    def _source_id_for(self, fname: str, src: str) -> str:
        src1_id, _spk1, src2_id, _spk2 = self._parse_mix_filename(fname)
        return src1_id if src == "s1" else src2_id

    def _read_wav(
        self, path: str, start: int = 0, stop: Optional[int] = None
    ) -> torch.Tensor:
        wav, _ = sf.read(path, start=start, stop=stop, dtype="float32")
        wav = torch.from_numpy(wav)
        if self.seg_len is not None and wav.shape[0] < self.seg_len:
            wav = torch.nn.functional.pad(wav, (0, self.seg_len - wav.shape[0]))
        return wav

    @staticmethod
    def _read_wav_with_pad(
        path: str, start: int, stop: Optional[int], pad_len: Optional[int]
    ) -> torch.Tensor:
        wav, _ = sf.read(path, start=start, stop=stop, dtype="float32")
        wav = torch.from_numpy(wav)
        if pad_len is not None and wav.shape[0] < pad_len:
            wav = torch.nn.functional.pad(wav, (0, pad_len - wav.shape[0]))
        return wav

    def _same_file_enrollment(
        self, audio: np.ndarray, avoid_start: int, avoid_stop: int, key: str
    ) -> np.ndarray:
        total_len = int(len(audio))
        intervals = []
        if avoid_start > 0:
            intervals.append((0, avoid_start, "left"))
        if avoid_stop < total_len:
            intervals.append((avoid_stop, total_len, "right"))
        intervals = [item for item in intervals if item[1] > item[0]]
        if not intervals:
            return np.zeros(self.enroll_len, dtype=np.float32)
        long_enough = [
            item for item in intervals if item[1] - item[0] >= self.enroll_len
        ]
        if long_enough:
            interval_idx = self._sample_offset(
                0, len(long_enough), key, "same_file_interval"
            )
            start, end, _side = long_enough[interval_idx]
            crop_start = self._sample_offset(
                start, end - self.enroll_len + 1, key, "same_file_crop"
            )
            return audio[crop_start : crop_start + self.enroll_len].astype(
                np.float32, copy=False
            )
        start, end, side = max(intervals, key=lambda item: item[1] - item[0])
        cropped = audio[start:end].astype(np.float32, copy=False)
        pad = self.enroll_len - len(cropped)
        if side == "left":
            return np.pad(cropped, (pad, 0)).astype(np.float32, copy=False)
        return np.pad(cropped, (0, pad)).astype(np.float32, copy=False)

    def _load_enrollment(
        self,
        current_fname: str,
        target_spk: str,
        target_src: str,
        mix_start: int,
        mix_stop: int,
    ) -> torch.Tensor:
        candidates = self.spk_to_refs[target_spk]
        src1_id, _spk1, src2_id, _spk2 = self._parse_mix_filename(current_fname)
        mixture_source_ids = {src1_id, src2_id}
        valid = []
        for cand_fname, cand_src in candidates:
            if cand_fname == current_fname:
                continue
            cand_src_id = self._source_id_for(cand_fname, cand_src)
            if cand_src_id in mixture_source_ids:
                continue
            valid.append((cand_fname, cand_src))
        if valid:
            choice = self._sample_offset(
                0, len(valid), current_fname, target_spk, "enroll_file"
            )
            enroll_fname, enroll_src = valid[choice]
        else:
            enroll_fname, enroll_src = (current_fname, target_src)
        enroll_path = os.path.join(self.data_dir, enroll_src, enroll_fname)
        audio, _ = sf.read(enroll_path, dtype="float32")
        audio = self._apply_enrollment_loudness_norm(audio, enroll_path)
        if enroll_fname == current_fname:
            audio = self._same_file_enrollment(
                audio, mix_start, mix_stop, current_fname
            )
            return torch.from_numpy(audio)
        if self.enroll_len is None:
            return torch.from_numpy(audio.astype(np.float32, copy=False))
        if len(audio) >= self.enroll_len:
            start = self._sample_offset(
                0,
                len(audio) - self.enroll_len + 1,
                current_fname,
                enroll_fname,
                "enroll_crop",
            )
            audio = audio[start : start + self.enroll_len]
        else:
            audio = np.pad(audio, (self.enroll_len - len(audio), 0))
        return torch.from_numpy(audio.astype(np.float32, copy=False))

    def _target_src_from_ref_path(self, ref_path: str) -> str:
        parent = Path(ref_path).parent.name
        if parent not in ("s1", "s2"):
            raise ValueError(f"Cannot infer target source from ref path: {ref_path}")
        return parent

    def _choose_scp_enrollment(
        self, key: str, mix_path: str, ref_path: str, aux_path: str
    ) -> Tuple[str, bool]:
        if self.enrollment_policy == "fixed":
            return (aux_path, True)
        src1_id, spk1, src2_id, spk2 = self._parse_mix_filename(
            os.path.basename(mix_path)
        )
        target_src = self._target_src_from_ref_path(ref_path)
        target_spk = spk1 if target_src == "s1" else spk2
        excluded_source_ids = {src1_id, src2_id}
        if self.enrollment_policy == "source_random":
            valid = []
            for entry_idx, cand_src_id in self.scp_spk_to_refs[target_spk]:
                cand_key, cand_mix_path, cand_ref_path, _cand_aux_path = self.entries[
                    entry_idx
                ]
                if cand_src_id in excluded_source_ids or cand_mix_path == mix_path:
                    continue
                valid.append((cand_key, cand_ref_path))
            if valid:
                choice = self._sample_offset(
                    0, len(valid), key, target_spk, "scp_source_enroll_file"
                )
                return (valid[choice][1], False)
            return (aux_path, True)
        aux_dir = os.path.dirname(aux_path)
        if aux_dir not in self._aux_dir_cache:
            self._aux_dir_cache[aux_dir] = sorted(
                (f for f in os.listdir(aux_dir) if f.endswith(".wav"))
            )
        excluded = {src1_id + ".wav", src2_id + ".wav"}
        candidates = [
            fname for fname in self._aux_dir_cache[aux_dir] if fname not in excluded
        ]
        if not candidates:
            return (aux_path, True)
        choice = self._sample_offset(0, len(candidates), key, "scp_enroll_file")
        return (os.path.join(aux_dir, candidates[choice]), True)

    def _resolve_scp_entry_index(self, idx: int) -> int:
        if not self.scp_group_by_mixture:
            return idx
        group = self.entry_groups[idx]
        if len(group) == 1:
            return group[0]
        if self.target_mode == "s1":
            for entry_idx in group:
                if self._target_src_from_ref_path(self.entries[entry_idx][2]) == "s1":
                    return entry_idx
            return group[0]
        if self.target_mode == "s2":
            for entry_idx in group:
                if self._target_src_from_ref_path(self.entries[entry_idx][2]) == "s2":
                    return entry_idx
            return group[-1]
        if self.target_mode in ("alternate", "both"):
            return group[idx % len(group)]
        if self.deterministic_eval:
            key = os.path.basename(self.entries[group[0]][1])
            return group[self._stable_int(key, "scp_group_target") % len(group)]
        return group[int(np.random.randint(0, len(group)))]

    def _load_scp_enrollment(
        self, key: str, path: str, apply_loudness_norm: bool = True
    ) -> torch.Tensor:
        audio, sr = sf.read(path, dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz, got {sr} Hz for {path}")
        if apply_loudness_norm:
            audio = self._apply_enrollment_loudness_norm(audio, path)
        if self.enroll_len is None:
            return torch.from_numpy(audio.astype(np.float32, copy=False))
        if len(audio) >= self.enroll_len:
            start = self._sample_offset(
                0,
                len(audio) - self.enroll_len + 1,
                key,
                os.path.basename(path),
                "enroll_crop",
            )
            audio = audio[start : start + self.enroll_len]
        else:
            audio = np.pad(audio, (0, self.enroll_len - len(audio)))
        return torch.from_numpy(audio.astype(np.float32, copy=False))

    def _getitem_scp(self, idx: int):
        idx = self._resolve_scp_entry_index(idx)
        key, mix_path, ref_path, aux_path = self.entries[idx]
        audio_len = self.lengths[idx]
        if self.seg_len is None:
            start, stop = (0, None)
        elif audio_len <= self.seg_len or not self.random_start:
            start, stop = (0, self.seg_len)
        else:
            start = self._sample_offset(
                0, audio_len - self.seg_len + 1, key, "mix_crop"
            )
            stop = start + self.seg_len
        enroll_path, apply_loudness_norm = self._choose_scp_enrollment(
            key, mix_path, ref_path, aux_path
        )
        mix = self._read_wav_with_pad(mix_path, start, stop, self.seg_len)
        target = self._read_wav_with_pad(ref_path, start, stop, self.seg_len)
        enroll = self._load_scp_enrollment(
            key, enroll_path, apply_loudness_norm=apply_loudness_norm
        )
        if self.dynamic_source_rebalance:
            mix, target = self._dynamic_rebalance_target_residual(mix, target)
        if self.normalize:
            mix_std = mix.std() + self.eps
            mix = (mix - mix.mean()) / mix_std
            target = (target - target.mean()) / mix_std
            enroll = (enroll - enroll.mean()) / (enroll.std() + self.eps)
        if self.waveform_scale != 1.0:
            mix = mix * self.waveform_scale
            target = target * self.waveform_scale
            enroll = enroll * self.waveform_scale
        return (mix, target, enroll)

    def _getitem_scp_pair(self, idx: int):
        group = self.entry_groups[idx]
        first_key, mix_path, _first_ref_path, _first_aux_path = self.entries[group[0]]
        audio_len = self.lengths[group[0]]
        pair_key = os.path.basename(mix_path)
        if self.seg_len is None:
            start, stop = (0, None)
        elif audio_len <= self.seg_len or not self.random_start:
            start, stop = (0, self.seg_len)
        else:
            start = self._sample_offset(
                0, audio_len - self.seg_len + 1, pair_key, "paired_mix_crop"
            )
            stop = start + self.seg_len
        mix = self._read_wav_with_pad(mix_path, start, stop, self.seg_len)
        loaded = []
        for entry_idx in group:
            key, cand_mix_path, ref_path, aux_path = self.entries[entry_idx]
            if cand_mix_path != mix_path:
                raise RuntimeError(
                    f"Mixed paths inside one scp target pair: {mix_path} vs {cand_mix_path}"
                )
            enroll_path, apply_loudness_norm = self._choose_scp_enrollment(
                key, mix_path, ref_path, aux_path
            )
            target = self._read_wav_with_pad(ref_path, start, stop, self.seg_len)
            enroll = self._load_scp_enrollment(
                key, enroll_path, apply_loudness_norm=apply_loudness_norm
            )
            loaded.append((target, enroll))
        if self.dynamic_source_rebalance:
            if len(loaded) != 2:
                raise RuntimeError(
                    "paired dynamic source rebalance requires exactly two targets"
                )
            mix, first_target, second_target = self._dynamic_rebalance_sources(
                loaded[0][0], loaded[1][0], mix
            )
            loaded = [(first_target, loaded[0][1]), (second_target, loaded[1][1])]
        samples = []
        for target, enroll in loaded:
            sample_mix = mix
            if self.normalize:
                mix_std = sample_mix.std() + self.eps
                sample_mix = (sample_mix - sample_mix.mean()) / mix_std
                target = (target - target.mean()) / mix_std
                enroll = (enroll - enroll.mean()) / (enroll.std() + self.eps)
            if self.waveform_scale != 1.0:
                sample_mix = sample_mix * self.waveform_scale
                target = target * self.waveform_scale
                enroll = enroll * self.waveform_scale
            samples.append((sample_mix, target, enroll))
        return samples

    def __getitem__(self, idx: int):
        if self.is_scp_mode:
            if self.scp_pair_targets:
                return self._getitem_scp_pair(idx)
            return self._getitem_scp(idx)
        file_idx = idx // 2 if self.target_mode == "both" else idx
        fname = self.files[file_idx]
        audio_len = self.lengths[file_idx]
        if self.seg_len is None:
            start, stop = (0, None)
        elif audio_len <= self.seg_len or not self.random_start:
            start, stop = (0, self.seg_len)
        else:
            start = self._sample_offset(
                0, audio_len - self.seg_len + 1, fname, "mix_crop"
            )
            stop = start + self.seg_len
        target_idx = self._target_index(idx, fname)
        target_src = "s1" if target_idx == 0 else "s2"
        _src1_id, spk1, _src2_id, spk2 = self._parse_mix_filename(fname)
        target_spk = spk1 if target_idx == 0 else spk2
        mix = self._read_wav(os.path.join(self.mix_dir, fname), start, stop)
        target = self._read_wav(
            os.path.join(self.data_dir, target_src, fname), start, stop
        )
        enroll = self._load_enrollment(
            fname,
            target_spk,
            target_src,
            mix_start=start,
            mix_stop=audio_len if stop is None else stop,
        )
        if self.dynamic_source_rebalance:
            mix, target = self._dynamic_rebalance_target_residual(mix, target)
        if self.normalize:
            mix_std = mix.std() + self.eps
            mix = (mix - mix.mean()) / mix_std
            target = (target - target.mean()) / mix_std
            enroll = (enroll - enroll.mean()) / (enroll.std() + self.eps)
        if self.waveform_scale != 1.0:
            mix = mix * self.waveform_scale
            target = target * self.waveform_scale
            enroll = enroll * self.waveform_scale
        return (mix, target, enroll)

    @staticmethod
    def collate_fn(batch):
        if batch and isinstance(batch[0], list):
            flat_batch = []
            for item in batch:
                flat_batch.extend(item)
            batch = flat_batch
        if len(batch[0]) not in (3, 4):
            raise ValueError(f"Expected 3- or 4-item samples, got {len(batch[0])}")
        mixes, targets, enrolls = zip(*[item[:3] for item in batch])
        lengths = [mix.shape[0] for mix in mixes]
        max_len = max(lengths)
        enroll_lengths = [enroll.shape[0] for enroll in enrolls]
        max_enroll_len = max(enroll_lengths)
        padded_mixes = []
        padded_targets = []
        padded_enrolls = []
        for mix, target in zip(mixes, targets):
            pad_len = max_len - mix.shape[0]
            if pad_len > 0:
                mix = torch.nn.functional.pad(mix, (0, pad_len))
                target = torch.nn.functional.pad(target, (0, pad_len))
            padded_mixes.append(mix)
            padded_targets.append(target)
        for enroll in enrolls:
            pad_len = max_enroll_len - enroll.shape[0]
            if pad_len > 0:
                enroll = torch.nn.functional.pad(enroll, (0, pad_len))
            padded_enrolls.append(enroll)
        collated = (
            torch.stack(padded_mixes, dim=0),
            torch.stack(padded_targets, dim=0),
            torch.stack(padded_enrolls, dim=0),
        )
        if len(batch[0]) == 3:
            return collated
        activities = [item[3] for item in batch]
        return collated + (torch.stack(activities, dim=0),)
