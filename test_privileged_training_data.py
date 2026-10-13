import numpy as np

from privileged_training_data import VARIANT_CONFIG, build_epoch_plan


def test_all_variants_cover_train_once_and_merge_singleton_tail():
    train = np.arange(513, dtype=np.int64)
    observed = np.ones(600, dtype=bool)
    for variant in VARIANT_CONFIG:
        plan = build_epoch_plan(train, observed, variant=variant, batch_size=256,
                                base_seed=9, evaluation_unit="iid", epoch=0)
        assert [len(batch) for batch in plan.batches] == [256, 257]
        joined = np.concatenate(plan.batches)
        assert len(np.unique(joined)) == len(train)
        assert set(joined.tolist()) == set(train.tolist())


def test_full_dropout_is_exact_and_natural_missing_never_becomes_observed():
    train = np.arange(10, dtype=np.int64)
    observed = np.asarray([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    full = build_epoch_plan(train, observed, variant="full", batch_size=4,
                            base_seed=9, evaluation_unit="iid", epoch=0)
    assert int(full.teacher_mask[train].sum()) == 6
    assert int(full.input_mask[train].sum()) == 3
    assert np.all(full.input_mask[6:] == 0)
    repeated = build_epoch_plan(train, observed, variant="full", batch_size=4,
                                base_seed=9, evaluation_unit="iid", epoch=0)
    assert np.array_equal(full.input_mask, repeated.input_mask)


def test_non_full_variants_do_not_drop_observed_dos():
    train = np.arange(8, dtype=np.int64)
    observed = np.asarray([1, 0, 1, 1, 0, 1, 1, 0], dtype=bool)
    for variant in ("base", "align", "distill", "align_distill"):
        plan = build_epoch_plan(train, observed, variant=variant, batch_size=3,
                                base_seed=9, evaluation_unit="iid", epoch=3)
        assert np.array_equal(plan.input_mask, plan.teacher_mask)


def test_ablation_variants_share_identical_epoch_batch_order():
    train = np.arange(40, dtype=np.int64)
    observed = np.ones(40, dtype=bool)
    plans = [
        build_epoch_plan(train, observed, variant=variant, batch_size=7,
                         base_seed=9, evaluation_unit="iid", epoch=2)
        for variant in VARIANT_CONFIG
    ]
    reference = np.concatenate(plans[0].batches)
    assert all(np.array_equal(np.concatenate(plan.batches), reference) for plan in plans[1:])
