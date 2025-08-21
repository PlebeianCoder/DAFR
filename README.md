# Text to Stealthy Adversarial Face Masks

This is the code implementation of "Text to Stealthy Adversarial Face Masks" by Ben Lewis, Thomas Moyse, James Parkinson, Elizabeth Telford, Callum Whitfield, Ranko Lazic.

## Diffusion Attacks against Facial Recognition (DAFR)

###  Installation

1. Create a Python virtual environment to install the requirements in freeze_requirements.txt, generated using pip freeze. We also provide a smaller requirements using pipreqs, but have not tested using these requirements.
2. You will also have to install locally into your virtual environment AdvDiffFace (advfaceutil) and PyFaceAR (pyfacear), two libraries created to provide utilities, benchmarks and the recognition frameworks. This can be done using pip too.
3. For prnet.pth and mobilefacenet_model_best.pth.tar, these can be downloaded from here https://github.com/AlonZolfi/AdversarialMask/tree/master and placed in the prnet/ and landmark_detection/pytorch_face_landmark/weights directories of each attack and in the benchmark processors within AdvDiffFace. Each attack is completely independent meaning there may be repeated code/weights, however one could change the path of the weights in each attack to point to one centralized weights. 
4. DLIB requires "shape_predictor_68_face_landmarks.dat" in AdvDiffFace/advfaceutil/recognition/processing which can be downloaded using the code in  DAFR/ffhq_align.py and then placed in the processing directory. Similarly, each part of the code is independent so the DLIB weights are too, but one could rewrite the paths to make this not so.
*Note that we used Python 3.8.18*

**For the differentiable rendering pipeline the following adjustment may need to be made to Kornia (https://github.com/AlonZolfi/AdversarialMask/issues/8).**

Some directory pathes are currently hardcoded such as in AdvDiffFace/advfaceutil/benchmark/statistic/cosineSuccess.py and CleanCode/AdvDiffFace/advfaceutil/benchmark/statistic/cosineStat.py

For weights they can be gathered as such:
* *FTED100*: We will release these weights at a later date
* *MFN*: https://github.com/HansSunY/DiffAM, then download the pretrained models and then find the MobileFaceNet weights
* *R100*: https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d, the backbone in ms1mv3_arcface_r100_fp16 
* *FaRL*: https://github.com/FacePerceiver/FaRL/blob/main/README.md, In the pretrained backbones section it is "FaRL-Base-Patch16-LAIONFace20M-ep16"

The DAFR directory contains the DAFR attack itself, but also benchmark code and other utilities.

* *mask_stats.py*: For calculating M-CMMD and T-CMMD, plus other mask utilities. **The CMMD currently in the benchmark measures the average individual CMMD (i.e the CMMD of each texture in a singleton with the reference in a singleton) and all CMMD statistics from the paper come from mask_stats.py being applied to the generated textures from benchmarks. This was experimented as a potential alternative.**
* *prepare_dataset.py*: Based on code from https://github.com/alonzolfi/adversarialmask, performs MTCNN alignment
* *ffhq_align.py*: Based on code from https://github.com/HansSunY/DiffAM, performs FFHQ alignment
* *BenchTexture.py*: performs the benchmark in the paper on DAFR
* *Txt2Adv3DAttack.py*: performs DAFR
* *ttrAttack.py*: Calls DAFR

AdvMask contains the code for the AdvMask attack (https://github.com/alonzolfi/adversarialmask) and SASMask contains the SASMask attack (https://github.com/small-seven/sasmask). The stable diffusion model used can be downloaded from https://huggingface.co/stabilityai/stable-diffusion-2-1-base.

For each of the attacks, they will have python code to run one instance of the attack *(DAFR: ttrAttack.py, SASMask: sas_attack.py, AdvMask: train.py)* so refer to this code to run the attacks, with DAFR in particular having an explanation of hyperparameters in DAFR/Txt2Adv3DAttack.py. Each benchmark takes command line arguments with their first being attack specific arguments (refer to the benchmark code for each attack for this) and then general benchmark ones found in AdvDiffFace/advfaceutil/benchmark/args.py.

During this project many different ideas were experimented with so a signficant proportion of this code is not used in the final paper, but could be useful to future work.

While most of the code was handwritten, LLMs have been used to assist development.

Acknowledgements:
```
@inproceedings{zolfi2023adversarial,
  title={Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models},
  author={Zolfi, Alon and Avidan, Shai and Elovici, Yuval and Shabtai, Asaf},
  booktitle={Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022, Grenoble, France, September 19--23, 2022, Proceedings, Part III},
  pages={304--320},
  year={2023},
  organization={Springer}
}

@article{gong2023stealthy,
  title={Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization},
  author={Gong, Huihui and Dong, Minjing and Ma, Siqi and Camtepe, Seyit and Nepal, Surya and Xu, Chang},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}

@InProceedings{Sun_2024_CVPR,
    author    = {Sun, Yuhao and Yu, Lingyun and Xie, Hongtao and Li, Jiaming and Zhang, Yongdong},
    title     = {DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24584-24594}
}

https://github.com/sayakpaul/cmmd-pytorch

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

https://github.com/deepinsight/insightface
```