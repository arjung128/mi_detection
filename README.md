# Deep Learning for Cardiologist-level Myocardial Infarction Detection in Electrocardiogram

This repository includes a PyTorch implementation of Deep Learning for Cardiologist-level Myocardial Infarction Detection in Electrocardiogram. Our code makes use of the Physikalisch-Technische Bundesanstalt (PTB) database, available [here](https://physionet.org/content/ptbdb/1.0.0/), and is based on the ConvNetQuake architecture, as described [here](https://advances.sciencemag.org/content/4/2/e1700578).

The architecture of the model is as follows:

![test_img](/architecture.png)

### Requirements
* Python 2.7
* NumPy 1.16.1 (or later)
* PyTorch 0.4.1 (or later)
* Matplotlib
* wfdb

#### Train the network

A reasonable set of hyperparameters is provided in `train.sh`. To train your own model:

```bash 
mkdir results
./train.sh
```
