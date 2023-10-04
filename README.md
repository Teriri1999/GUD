# GAN-based unsupervised SAR despeckling and editing
This repository contains the code implementation of the paper "Exploring Latent Space of GAN for SAR Despeckling and Editing."

## Usage
### Train GUD model
Run run_train.py to train the GUD (GAN-based Unsupervised Despeckling) model.
```python
python run_train.py --gan_weights ../weight.pth --out out_path
```

### Implement SAR despeckling
Execute GUD_visualization.py to generate sample editing results.
```python
python GUD_visualization.py
```
