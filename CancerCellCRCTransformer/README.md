# Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study
<div style="display: flex; align-items: center;">
    <p style="margin-right: 50px;"><b>Abstract:</b> Deep learning (DL) can accelerate the prediction of prognostic biomarkers from routine pathology slides in colorectal cancer (CRC). However, current approaches rely on convolutional neural networks (CNNs) and have mostly been validated on small patient cohorts. Here, we develop a new transformer-based pipeline for end-to-end biomarker prediction from pathology slides by combining a pre-trained transformer encoder with a transformer network for patch aggregation. Our transformer-based approach substantially improves the performance, generalizability, data efficiency, and interpretability as compared with current state-of-the-art algorithms. After training and evaluating on a large multicenter cohort of over 13,000 patients from 16 colorectal cancer cohorts, we achieve a sensitivity of 0.99 with a negative predictive value of over 0.99 for prediction of microsatellite instability (MSI) on surgical resection specimens. We demonstrate that resection specimen-only training reaches clinical-grade performance on endoscopic biopsy tissue, solving a long-standing diagnostic problem. </p>
    <img src="visualizations/2023-07_graphical-abstract.jpg" alt="Image Alt Text" width="400px" align="right"/>
</div>


<!-- ![pipeline](2023-07_graphical-abstract.jpg) -->

## Folder structure
In  this folder you find additional resources used for the publication that are not contained in the main repository:
* `trained_models`: Folder with our models trained on the multicentric dataset for MSI high, BRAF, and KRAS prediction
    * `BRAF_CRC_model.pth`
    * `KRAS_CRC_model.pth`
    * `MSI_high_CRC_model.pth`
* `visualizations`: Folder with resourcesd for the visualizations in the publication
    * `Visualize_results.ipynb`: Jupyter notebook to visualize the results and plot the data overview figure
    * `UserStudy.ipynb`: Jupyter notebook to evaluate the user study
* `config.yaml`: Config file used for training the models
* `evaluations_user_study.xlsx`: Table with expert evaluations of high attention tiles from the user study
* `experimental_results.xlsx`: Table with all experimental results
* `train_num_samples.py`: Script to train models with different numbers of training samples
