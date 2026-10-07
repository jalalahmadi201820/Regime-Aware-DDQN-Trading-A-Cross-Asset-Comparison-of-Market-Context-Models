# Regime-Aware DDQN Trading Code

This folder contains the two experiment scripts used for the paper:

- `BTC_ETH_HMM.py` runs Bitcoin and Ether.
- `DJIA_HMM.py` runs the DJIA30 equal-weighted composite.

Both scripts expect the input data files next to the script:

- `BTCUSD_daily.pkl`
- `ETHUSD_daily.pkl`
- `djia30_full_2009_2022.csv`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python BTC_ETH_HMM.py
python DJIA_HMM.py
```
