# Adversarially Learned Transformations (ALT)
Code provided as part of supplementary material for Paper ID 11557.
To reproduce results for each benchmark, the following steps should be followed.

First, download the data. Since supplementary material is capped at 100MB, we are providing instructions for each dataset
1. Digits -- data will download automatically if you run `run_alt_mnist.sh`
2. PACS -- download from https://mega.nz/#F!jBllFAaI!gOXRx97YHx-zorH5wvS6uw
3. OfficeHome can be downloaded from the official release https://www.hemanthdv.org/officeHomeDataset.html

Digits:
`bash run_alt_mnist.sh`
PACS:
`bash run_alt_pacs.sh`
Office-Home:
`bash run_alt_officehome.sh`

The code structure is inspired from RandConv https://github.com/wildphoton/RandConv and also borrows some features from Sagnet https://github.com/hyeonseobnam/sagnet .