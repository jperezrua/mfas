[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mfas-multimodal-fusion-architecture-search-1/action-recognition-in-videos-on-ntu-rgbd)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ntu-rgbd?p=mfas-multimodal-fusion-architecture-search-1)

# MFAS: Multimodal Fusion Architecture Search


## This code

This is an implementation of the paper:

```
@inproceedings{perez2019mfas,
  title={Mfas: Multimodal fusion architecture search},
  author={P{\'e}rez-R{\'u}a, Juan-Manuel and Vielzeuf, Valentin and Pateux, St{\'e}phane and Baccouche, Moez and Jurie, Fr{\'e}d{\'e}ric},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6966--6975},
  year={2019}
}
```

## Usage

We focus on the NTU experiments in this repo. The file `main_found_ntu.py` is used to train and test architectures that were already found. Pick one of them by using the `--conf N` argument. This script should be easy to modify if you want to try other architectures.

Our best found architecture on NTU is slightly different to the one reported in the paper,
it can be tested like so:

`
python main_found_ntu.py --datadir ../../Data/NTU --checkpointdir ../../Data/NTU/checkpoints/ --use_dataparallel --test_cp best_3_1_1_1_3_0_1_1_1_3_3_0_0.9134.checkpoint --conf 4 --inner_representation_size 128 --batchnorm
`

To test the architecture from the paper, you can run:

`
python main_found_ntu.py --datadir ../../Data/NTU --checkpointdir ../../Data/NTU/checkpoints/ --use_dataparallel --test_cp conf_[[3_0_0]_[1_3_0]_[1_1_1]_[3_3_0]]_both_0.896888457572633.checkpoint
`

Of course, set your own Data and Checkpoints directories.

## Download the pretrained checkpoints

We provide pretrained backbones for RGB and skeleton modalities as well as some pretrained found architectures in here: [Google Drive link](https://drive.google.com/open?id=1wcIepkmCf2NRfnhXVdoNu6wSxkpZmMNm)



