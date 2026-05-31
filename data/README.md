# Data

Large raw datasets are not included in this repository. Generate or download data with the provided scripts:

```bash
make data-synth
make generate-real DATASET=electricity
make generate-real DATASET=exchange
make generate-real DATASET=metrla
```

Processed datasets are written under `data/interim/<dataset>/`. See `docs/datasets.md` for sources, preprocessing notes, and redistribution constraints.

