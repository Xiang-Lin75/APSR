"""Canonical paired-target dataset settings for the paper recipes."""

from .dataset import WSJ02MixTSEDataset


def build_dataset(manifest_dir, training=False, dynamic_source_rebalance=False):
    return WSJ02MixTSEDataset(
        manifest_dir=str(manifest_dir),
        dataset_type="scp",
        sample_rate=8000,
        segment=4.0 if training else None,
        enroll_segment=3.0 if training else None,
        normalize=False,
        waveform_scale=1.0,
        random_start=training,
        deterministic_eval=not training,
        target_mode="random",
        enrollment_policy="directory_random" if training else "fixed",
        enrollment_loudness_norm=False,
        scp_group_by_mixture=False,
        scp_pair_targets=True,
        drop_short=False,
        dynamic_source_rebalance=dynamic_source_rebalance if training else False,
        dynamic_sir_min_db=-3.0,
        dynamic_sir_max_db=3.0,
        dynamic_rebalance_probability=1.0,
        dynamic_rebalance_peak_limit=0.9,
    )
