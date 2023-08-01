# idkidc

Training script for multiple instance learning with various aggregation models. Based on Pytorch Lightning. All aggregation modules can be loaded as model files in the classifier lightning module. Loss, models, optimizers, and schedulers can be specified as strings according to the PyTorch name in the config file. Have a look at the `get_xyz` functions for more details on this.

## Repository structure
```
├── models                          # all model files
│   ├── aggregators
│   │   ├── __init__.py  
│   │   ├── aggregator              # base class for aggregation modules 
│   │   ├── attentionmil.py         # model by Ilse et al., 2018
│   │   ├── lamil.py 
│   │   ├── model_utils.py          # common layers and functions used in the other models
│   │   ├── perceiver.py
│   │   ├── test_aggregators.py     # test new aggregators
│   │   ├── transformer.py
│   │   ├── transmil.py             # model by Shao et al. 2021
│   ├── histaugan                   # model from Sophia for augmentation
│   │   ├── __init__.py  
│   │   ├── augment.py  
│   │   ├── model.py  
│   │   ├── networks.py  
├── classifier.py                   # lightning module for feature classification
├── config.yaml                     # config file for training arguments
├── data.py
├── environment.yaml                # config file for conda environment
├── Hackathon.ipynb                 # notebook for running the code in google colab
├── main.py                         # train and test models -- should be split up
├── options.py                      # argument parsing, overrides config file when arguments are given
├── run.sh
├── utils.py                        # get files and other utils
```

## Contribute

There are multiple tods left in the code (marked with TODO). Feel free to help solving them.

## Environment

Tipp: Install mamba for faster package installation in your conda base environmend (`conda install -c conda-forge mamba`)

Then install the following packages needed for a minimal working environement:
* mamba/conda: `pytorch pytorch-lightning wandb einops pyparsing h5py pandas`
* pip: `dgl`

Alternatively, install my conda env from `environment.yaml`:
```sh
conda env create --file environment.yaml
```

## Data structure

* `clini_table.xlsx`: Table (Excel-file) with clinically important labels. Should have the following format (or similar), each patient has a unique entry .

| PATIENT	| TARGET	| GENDER	| AGE |
| ---       | ---       | ---       | --- |
| ID_345    | positive	| female	| 61  |
| ID_459    | negative	| male	    | 67  |
| ID_697    | NA	    | female	| 42  |

* `slide.csv`: Table (csv-file) with patient id's matched with slide / file names. Patients can have multiple entries if they have multiple slides.

| FILENAME	| PATIENT	|
| ---       | ---       |
| ID_345_slide01    | ID_345    |
| ID_345_slide02    | ID_345    |
| ID_459_slide01    | ID_459    |

* folder with features as `.h5-files`. Filenames correspond to filenames in `slide.csv`

