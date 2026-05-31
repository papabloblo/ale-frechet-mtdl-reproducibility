# Results

`results/published/` contains the aggregated reference outputs shipped with the paper submission. Raw logs, checkpoints, and exploratory runs are intentionally excluded.

When reproducing the experiments, new outputs are written under:

```text
results/comparisons/<dataset>/
```

Compare reproduced aggregate CSVs against `results/published/` at the level of method rankings, mean/std metrics, and reported trends rather than exact floating-point equality.

Use:

```bash
make compare-to-published
```

to verify that the bundled published outputs can rebuild the main paper CSVs and tables in a clean clone. Use:

```bash
make validate-published
```

after a full experimental rerun and `make paper-results` to compare regenerated paper CSVs with the bundled reference CSVs.

