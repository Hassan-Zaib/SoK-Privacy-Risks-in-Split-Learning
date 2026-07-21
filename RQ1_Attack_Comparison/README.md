# RQ1 Attack Comparison

RQ1 compares representative Split Learning Interface Attack families under matched two-party split-learning settings.

Environment

Use the splitlearn Conda environment.

Run

```bash
conda activate splitlearn
bash run_all.sh
```

## Full Reproduction

`run_all.sh` launches the full RQ1 experiment set across attacks, datasets, and split points. It may take several hours and requires the datasets described in `../docs/dataset_setup.md`.

Code Layout

code/unsplit contains UnSplit experiments.
code/sdar contains SDAR experiments.
code/gan_dra contains GAN-DRA and MAERA experiments.
code/simlia contains SimLIA experiments.

Results

Final processed metrics and qualitative reconstructions are stored in results.
