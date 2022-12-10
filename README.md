# Adversarially Learned Transformations (ALT)
Code for our paper: ["Improving Diversity with Adversarially Learned Transformations for Domain Generalization"](https://arxiv.org/abs/2206.07736) (WACV 2023).
To reproduce results for each benchmark, the following steps should be followed.

## Data Download:
First, download the data using the following instructions:
1. Digits -- data will download automatically if you run `run_alt_mnist.sh`
2. PACS -- download from https://mega.nz/#F!jBllFAaI!gOXRx97YHx-zorH5wvS6uw
3. OfficeHome can be downloaded from the official release https://www.hemanthdv.org/officeHomeDataset.html

## Train and Evaluate
- Digits: `bash run_alt_mnist.sh`
- PACS: `bash run_alt_pacs.sh`
- Office-Home: `bash run_alt_officehome.sh`

## Acknowledgements
Part of the code structure is borrowed from RandConv https://github.com/wildphoton/RandConv and Sagnet https://github.com/hyeonseobnam/sagnet .
We thank the authors of these papers.
