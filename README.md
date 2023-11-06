# Tracking SLAM

This folder contains the code to run the SLAM algorithm and generate dense reconstructions.

## Installing 

To install the pipeline navigate to Tracking_SLAM/CoTrackSLAM and run

```
source install.sh
```

This will install a conda enviroment named `TrackingSLAM` as well as [g2opy](https://github.com/uoip/g2opy). Note that g2opy requires `cmake` to be installed. The script was tested on Ubuntu 20.04.6 LTS.



## Running

To run on your data, setup a directory as follows

```
└── data
    ├── images
    │   ├── 00000000.jpg
    │   ├── 00000001.jpg
    │   ├── 00000002.jpg
    │   └── ...
    ├── calibration.json
    ├── [mask.bmp]
    └── [poses_gt.txt]
```

For the structure of the `calibration.json` file, please view the example provided in `data_example`. Note that while the intrinsics are required, the FPS is currently not used.
The poses are required to be in [TUM format](https://github.com/MichaelGrupp/evo/wiki/Formats#tum---tum-rgb-d-dataset-trajectory-format) with the frame index being used as the timestamp. The poses are assumed to be the Camera-To-World transformations.

After setting up the data directory, navigate to `run_slam.py` and set `data_root` to the path of your data directoy. Moreover, define the frames to be used by setting `start_idx` and `end_idx` to the first and last frame indices to be included.

Then, to start the slam pipeline run

```
python run_slam.py
```

After the run is finished, the output+visualizations will be saved in the folder `experiments/TIMESTAMP`. Note that currently creating the point trajectory visualization takes very long if tracking a lot points.