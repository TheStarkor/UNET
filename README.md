# UNET
Modified U-Net([MICCAI 2015](https://arxiv.org/abs/1505.04597)) for refining the approximated stereo images by utilizing SloMo's flow approximated network([CVPR 2018](https://arxiv.org/abs/1712.00080))

## Stereo Image Generation
### Bilinear Sampler(Input Data)
Using bilinear sampler in monodepth network([CVPR 2017](https://arxiv.org/abs/1609.03677))

- Results  
Code : [colab](https://colab.research.google.com/drive/1XMEdilKSPpxdYobQnIfAXcCvQdNCS34F?usp=sharing)  

![res1](https://user-images.githubusercontent.com/45455072/84602140-253e1180-aec0-11ea-8aa1-41d938dc317b.gif)  
PSNR = 15.848

![res2](https://user-images.githubusercontent.com/45455072/84602143-28390200-aec0-11ea-8819-5c75f4c62508.gif)  
PSNR = 15.896
