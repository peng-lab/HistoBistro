# idkidc

Preparation: 

    CTransPath: 
        224x224 pixels
        weights: https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view
        paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043 
        repo: https://github.com/Xiyue-Wang/TransPath

    RetCCL:
        256x256 pixels @ multiple scales
        weights: https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL
        paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841522002730 
        repo: https://github.com/Xiyue-Wang/RetCCL


    Kimianet: 
        1000x1000 pixels @ 20x magnification
        weights: https://kimialab.uwaterloo.ca/kimia/index.php/sdm_downloads/kimianet-weights/ 
        paper: https://arxiv.org/abs/2101.07903
        Code Samples: https://kimialab.uwaterloo.ca/kimia/index.php/sdm_downloads/kimianet-feature-extraction-code-samples/ 


create your conda environment: 
conda create --name feature_ex python=3.9

activate conda env: 
conda activate feature_ex

install all dependencies:
pip install -r requirements.txt

Change the model paths in model/model.py to where you stored the weights.
Start the feature extraction by calling the feature extraction skript feature_extraction.py, e.g. in the commandline
python feature.py --slide_path /path/to/slides --save_path /path/to/save --file_extension .czi --models kimianet --scene_list 0 1 --save_patch_images True --patch_size 256 --white_thresh 170 --black_thresh 0 --invalid_ratio_thresh 0.5 --edge_threshold 4 --resolution_in_mpp 0 --downscaling_factor 8 --BGR_to_RGB True --save_tile_preview True --preview_size 4096


Comments: 
There is maybe still some speedup possible. No benchmarking was done between different libraries used to load slides.
The attempt in multithreading lead to CUDA errors. Not sure if there is something to gain there though.. Most time takes the loading of the slides and scenes, not sure if doing that in parallel is faster or if RAM/harddrive reading is the bottleneck. 

