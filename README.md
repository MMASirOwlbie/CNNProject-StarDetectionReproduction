Star Detection & Recognition System using CNNs (Reproduction)

Reproduction by Muhammad Muneeb Ahmed (31588) and Ushna Jalil (30911).
This repository is a reproduction of the `ELUnet` star detection and sub-pixel centroid regression pipeline based on HongruiZhao/CNNStarDetectCentroid, with a generated dataset of synthetic star images simulated over stray-light dark frames.


## Technical Choices
We implemented modernized backwards compatibility, fixing a dependency issue in CentroidNet.py by migrating legacy imports (from numpy.lib.arraypad import pad) to the modern top-level API (from numpy import pad).

Also replaced single-directory creation calls (os.mkdir) with recursive creation (os.makedirs(exist_ok=True)) to prevent path collision errors during clean runs (also I coulnd't get it to work otherwise. x_x) 


1. Environment Setup
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/CNNStarDetectCentroid-Reproduction.git](https://github.com/YOUR_GITHUB_USERNAME/CNNStarDetectCentroid-Reproduction.git)
cd CNNStarDetectCentroid-Reproduction
pip install -r requirements.txt
```

2. Dataset Generation
```bash
cd data_generation
python download_darkframes.py  # Downloads stray light dark frame archives
python main_generate_data.py
```
Note that for this, we generated 500 total images, with 300 for training, and 100 each for both val and testing. 
Did not perform night-sky testing on this; purely pulled metrics and compared accuracy, precision and F1 score from the official paper. (Specifically Tables II and III from Section IV. Experiments).

3. Model Training & Monitoring
```bash
cd ../training
python training_stepLR.py --ep 30 --batch_size 1 --trial 1
```

## Results & Artifacts
* **Loss Curves:** Saved in `training/runs/`
* **Model Checkpoints:** Saved in `training/models/`
