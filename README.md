# Syn2RealTrack: Bridging the Gap Between Synthetic and Real-World Datasets for Online Multi-View Multi-Target Tracking

### Team SKKU-AL-T1 - ID 34

Solution for 2026 AI City Challenge Track 1: Multi-Camera 3D Perception (Sim2Real)

---
## I. Dataset preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2026-track1/

Download dataset to the folder **<MTMC_Tracking_2026>**

The dataset folder structure should be as following:

```shell
<MTMC_Tracking_2026>
│   ├── train
│   │   ...
│   ├── val
│   │   ...
│   ├── test
│   │   ├── Warehouse_023
│   │   │   ├── videos
│   │   │   ├── calibration.json
│   │   │   └── map.png
│   │   ├── Warehouse_024
│   │   ├── Warehouse_025
│   │   └── Warehouse_026
```

---
## II. Environment setup

#### a. Installation MinicondaMinicondaMini|Anaconda and FFmpeg:

Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

Install FFmpeg: https://www.ffmpeg.org/

#### b. Create conda environment:

Follow the instructions in the following files to install the required dependencies.

```shell
conda create --name Syn2RealTrack python=3.12 -y

conda activate Syn2RealTrack

bash setup.sh
```

#### c. Load weights:

Download each weight file and move them into the corresponding zoo folder:

```shell
zoo
├── detection
│   ├── rfdetr_aicity_wh023_2xlarge_r1920_b4_e300
│   ├── rfdetr_aicity_wh024_2xlarge_r1920_b4_e300
│   ├── rfdetr_aicity_wh026_2xlarge_r1920_b4_e200
│   ├── rfdetr_aicity_wh027_2xlarge_r1920_b4_e200
│   ├── yolo26x_aic26_025_1920
└── reidentification
    ├── kpr
    ├── SOLIDER
    └── vitpose-plus-huge
```

---
## III. Inference

#### a. Adjust the configuration

Default link to the dataset in the code is:

```ymal
data:
  root: &dataset ".../MTMC_Tracking_2026"
...
data_writer:
  root: ".../MTMC_Tracking_2026_processing_baseline"
```

".../MTMC_Tracking_2026" is the <input_folder>.

".../MTMC_Tracking_2026_processing_baseline" is the <output_folder>.

You update to change it to your own path in each config file.

```shell
configs
│   ├── warehouse_023.yaml
│   ├── warehouse_024.yaml
│   ├── warehouse_025.yaml
│   ├── warehouse_026.yaml
│   └── warehouse_027.yaml
```

#### b. Run the code

Run the following commands in the terminal:

```shell
bash run_pipline.sh
```

After running all command above, the output files will be in the folder 

```shell
<output_folder>/mots_multi/track1.txt
```

---
## IV. Citation

```
Updating soon.

Accepted in AI City Challenge Workshop in ECCV 2026
```

---
## V. Acknowledgement

Most of the code is adapted from [Mon](https://github.com/phlong3105/mon).

This repository also features code from
[Ultralytics](https://github.com/ultralytics/ultralytics),
[RF-DETR](https://github.com/roboflow/rf-detr),
[segment-anything](https://github.com/facebookresearch/segment-anything),
[Torchreid](https://github.com/kaiyangzhou/deep-person-reid),
[KPR](https://github.com/vlsomers/keypoint_promptable_reidentification),
and [Bot-Sort](https://github.com/NirAharon/BoT-SORT)
