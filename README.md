# Regime-Aware DDQN Trading Code

This folder contains the two experiment scripts used for the paper:

- `HMM.py` runs Bitcoin and Ether.
- `dji_hmm.py` runs the DJIA30 equal-weighted composite.

Both scripts expect the input data files next to the script:

- `BTCUSD_daily.pkl`
- `ETHUSD_daily.pkl`
- `djia30_full_2009_2022.csv`

The scripts print only one `tqdm` progress bar and the final paper table. They do not write CSV, pickle, image, or chart outputs.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python HMM.py
python dji_hmm.py
```
