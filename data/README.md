# Dataset Download Instructions

This project uses the **WESAD** dataset (Schmidt et al., ICMI 2018).
Official page: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

1. Download `WESAD.zip` (2.25 GB, direct link, no form):
   https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
2. Extract it into the project root so you get `stress-detection/WESAD/S2/S2.pkl`, etc.
   Only the `S*.pkl` files are used:

```bash
wget -O WESAD.zip "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
unzip WESAD.zip 'WESAD/S*/S*.pkl' -d .
```

3. Build the windowed dataset (writes `data/processed/wesad_100hz_60s_step15.npz`):

```bash
python -m src.data
```

> On Google Colab, `notebooks/04_colab_train.ipynb` does all of this automatically.
