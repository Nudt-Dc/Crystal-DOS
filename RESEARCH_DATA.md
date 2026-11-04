# Crystal-DOS research data

Source data for *Electronic-structure Knowledge Transfer for Label-efficient Prediction of Crystal Elasticity*.

Version: 2026-11-04. Core implementation: https://github.com/Nudt-Dc/Crystal-DOS

## Files and interpretation

All CSV files preserve the saved numerical results without recalculation.

| File | Rows | Contents |
| --- | ---: | --- |
| validation_run_metrics.csv | 486 | Validation metrics for each training resource, partition, label fraction, seed and condition. |
| paired_target_deltas.csv | 162 | Paired-DOS minus structure-only K and G MAE for matched runs. Negative values favor paired DOS. |
| potnet_validation_plot_values.csv | 48 | PotNet minus comparator MAE. Negative values favor PotNet. Training windows remain separate. |
| external_all_72_groups.csv | 72 | All external evaluation groups, including run counts, failures, descriptive means and run standard deviations. |
| external_primary_6_groups.csv | 6 | Matched primary external comparisons for two training resources and three label fractions, each with both targets and saved confidence intervals. |

K denotes bulk modulus and G denotes shear modulus. MAE and its differences are in GPa. Selection scores are the saved model-selection quantities, distinct from GPa MAEs. Resource identifies the training dataset (MP or AFLOW); fraction identifies the labeled training proportion (1%, 5%, or 10%). Fold and seed identify related fitted models. The validation files support the low-label, strong-baseline and target-consistency figures; they are distinct from the external evaluation.

## External evaluation

The external source is the Materials Cloud MC3D mechanical extension. Of 424 frozen structure candidates, 82 materials had eligible targets, 340 lacked the specified branch and two had nonpositive targets. The full comparison planned 1,944 runs, with 1,942 successes and two execution failures. Both failures affect AFLOW 1% auxiliary single-loss conditions. Their complete-27-run fields are blank, while their 26-run descriptive results are retained. All primary comparisons have 27 successful runs.

Material count is distinct from run count. The 27 partition-seed models evaluate the same external materials. Run standard deviations describe variation across these related models. The saved primary confidence intervals use 2,000 material-bootstrap replicates, seed 20260929, percentile endpoints, and no multiplicity adjustment. These endpoints are copied from the saved results.

## Scope and provenance

This release provides author-generated numerical study results and figure source data. It contains no raw crystal structures, DOS spectra, material-ID rosters, training partitions, control masks, source target tables or model checkpoints. It supports inspection and replotting of the supplied results; per-material predictions and targets needed to recompute the bootstrap are not included.

The CSVs are byte-identical copies of the previously verified study exports and figure ledgers. SHA256SUMS.txt records the release checksums. Source databases retain their own access and reuse conditions; this package does not redistribute their raw datasets or grant rights to them. No additional reuse license is assigned by this release.
