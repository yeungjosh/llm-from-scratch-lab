"""Phase 0 smoke tests — modules import, estimator math is sane."""

from __future__ import annotations


def test_lfslab_imports():
    import lfslab  # noqa: F401
    import lfslab.data  # noqa: F401
    import lfslab.estimator  # noqa: F401
    import lfslab.generate  # noqa: F401
    import lfslab.hardware  # noqa: F401
    import lfslab.kv_cache_toy  # noqa: F401
    import lfslab.model  # noqa: F401
    import lfslab.plotting  # noqa: F401
    import lfslab.profile_train  # noqa: F401
    import lfslab.tokenizer  # noqa: F401
    import lfslab.train  # noqa: F401


def test_param_count_matches_manual_calc():
    from lfslab.estimator import ModelSpec, param_count

    m = ModelSpec(
        d_model=128,
        d_ff=512,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        head_dim=32,
        vocab_size=1000,
        tied_embeddings=True,
    )
    per_layer = 3 * 128 * 512 + 2 * 128 * (4 + 4) * 32 + 128
    embed = 128 * 1000
    assert param_count(m) == 2 * per_layer + embed


def test_lr_schedule_warmup_and_decay():
    from lfslab.train import get_lr_cosine_schedule

    assert get_lr_cosine_schedule(0, 1.0, 0.1, 10, 100) < 1.0
    assert get_lr_cosine_schedule(10, 1.0, 0.1, 10, 100) == 1.0
    assert get_lr_cosine_schedule(200, 1.0, 0.1, 10, 100) == 0.1


def test_hardware_catalogue_has_mps_and_a100():
    from lfslab.hardware import CATALOGUE

    assert "mps_m3_max" in CATALOGUE
    assert "a100_80g" in CATALOGUE
    assert CATALOGUE["a100_80g"].peak_flops_per_s > 0
