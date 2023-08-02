# HistoBistro

Pipeline for weakly-supervised learning on histology images. The pipeline contains various models for multiple instance learning with different aggregation models. Based on Pytorch Lightning. All aggregation modules can be loaded as model files in the classifier lightning module. Loss, models, optimizers, and schedulers can be specified as strings according to the PyTorch name in the config file.

![](CancerCellCRCTransformer/model.png)

## Repository structure
```
├── models                          # all model files
│   ├── aggregators
│   │   ├── __init__.py  
│   │   ├── aggregator              # base class for aggregation modules 
│   │   ├── attentionmil.py         # model by Ilse et al., 2018
│   │   ├── lamil.py                # model by Reisenbüchler er al., 2022
│   │   ├── model_utils.py          # common layers and functions used in the other models
│   │   ├── perceiver.py
│   │   ├── test_aggregators.py     # test new aggregators
│   │   ├── transformer.py
│   │   ├── transmil.py             # model by Shao et al. 2021
│   ├── histaugan                   # model from Wagner et al., 2021, for stain augmentation
│   │   ├── __init__.py  
│   │   ├── augment.py  
│   │   ├── model.py  
│   │   ├── networks.py  
├── classifier.py                   # lightning module for feature classification
├── config.yaml                     # config file for training
├── data.py                         # dataset class and helper functions
├── environment.yaml                # config file for conda environment
├── main.py                         # train and test models
├── options.py                      # argument parsing, overrides config file when arguments are given
├── run.sh
├── utils.py                        # get files and other utils
```

## Setup

Setup `data_config.yaml` and `config.yaml` with your data paths and training configurations. All entries in brackets `< >` should be customized.

Install the following packages needed for a minimal working environement:
* mamba/conda: `pytorch pytorch-lightning wandb einops pyparsing h5py pandas`
* pip: `dgl`

Alternatively, install the conda env from `environment.yaml`:
```sh
conda env create --file environment.yaml
```

## Data structure

* `clini_table.xlsx`: Table (Excel-file) with clinically important labels. Each patient has a unique entry, column names `PATIENT` and `TARGET` are required.

| PATIENT	| TARGET	| GENDER	| AGE |
| ---       | ---       | ---       | --- |
| ID_345    | positive	| female	| 61  |
| ID_459    | negative	| male	    | 67  |
| ID_697    | NA	    | female	| 42  |

* `slide.csv`: Table (csv-file) with patient id's matched with slide / file names (column names `FILENAME` and `PATIENT`). Patients can have multiple entries if they have multiple slides.

| FILENAME	| PATIENT	|
| ---       | ---       |
| ID_345_slide01    | ID_345    |
| ID_345_slide02    | ID_345    |
| ID_459_slide01    | ID_459    |

* folder with features as `.h5-files`. Filenames correspond to filenames in `slide.csv`


## Training and testing

You can train your model on a multi-centric dataset with the following k-fold cross validation (k=5) scheme where `--` (train) `**` (val), and `##` (test).
```
[--|--|--|**|##]
[--|--|**|##|--]
[--|**|##|--|--]
[**|##|--|--|--]
[##|--|--|--|**]
```

by running 
```
python train_k-fold.py --name <name> --data_config <path/to/data_config.yaml> --config <path/to/config.yaml>
```
and test it on the in-domain test set and external cohorts by running
```
python test.py --name <name> --data_config <path/to/data_config.yaml> --config <path/to/config.yaml>
```

## Publications

Information to publications based on this repository are grouped in the respective folders. Find more detailed information in the README.md files in the respective folders.

If you consider this useful for your research, please cite the following preprint:
```
@misc{wagner2023fully,
      title={Fully transformer-based biomarker prediction from colorectal cancer histology: a large-scale multicentric study}, 
      author={Sophia J. Wagner and Daniel Reisenbüchler and Nicholas P. West and Jan Moritz Niehues and Gregory Patrick Veldhuizen and Philip Quirke and Heike I. Grabsch and Piet A. van den Brandt and Gordon G. A. Hutchins and Susan D. Richman and Tanwei Yuan and Rupert Langer and Josien Christina Anna Jenniskens and Kelly Offermans and Wolfram Mueller and Richard Gray and Stephen B. Gruber and Joel K. Greenson and Gad Rennert and Joseph D. Bonner and Daniel Schmolze and Jacqueline A. James and Maurice B. Loughrey and Manuel Salto-Tellez and Hermann Brenner and Michael Hoffmeister and Daniel Truhn and Julia A. Schnabel and Melanie Boxberg and Tingying Peng and Jakob Nikolas Kather},
      year={2023},
      eprint={2301.09617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}```


