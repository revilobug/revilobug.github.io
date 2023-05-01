## Intro
The paper "Not Just Streaks: Towards Ground Truth for Single Image Deraining'' by Yunhao Ba, Howard Zhang, Ethan Yang, Akira Suzuki, Arnold Pfahnl, Chethan Chinder Chandrappa, Celso M. de Melo, Suya You, Stefano Soatto, Alex Wong, and Achuta Kadambi addresses the problem of single image deraining, where the goal is to remove rain streaks from a single input image. 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/61512660/235407839-e6873009-cb66-417c-8f63-0567f662b5d1.png">

Heavy rainfall can cause degradation of image quality and obfuscation of visual information. Many important downstream 
visual tasks such as object detection or semantic segmentation become intractable with the introduction of rain artifacts. In automated driving, for instance, the inability to complete object detection tasks can have fatal consequences. The development of effective de-raining methods will be useful for industrial applications as well as in image restoration research.

The task of rain removal is formulated as the equation $$𝑂=𝐵+𝑅$$ where $O$ represents some input image with rain $B$ represents the background layer, and $R$ represents the rain artifact layer. The task of de-raining involves finding a mapping from $O$ to $B$.

However, single image de-raining is inherently an ill-posed problem. Since raindrops cause information loss, reconstructing what exists behind a raindrop is a generative task. A unique, perfect solution, $B$, cannot be derived from $O$. It is impossible to reconstruct with certainty image information destroyed by rainfall. Therefore, any mapping from $O$ to $B$ is at best an approximation.

Traditional de-raining methods utilize image priors. For example, one solution involved decomposing a rainy input image into low and high frequency components to identify rain streaks [[1]](#1). However, these techniques lack generalizability and are ineffective at completely removing rain.

Deep learning has recently emerged as the preferred de-raining method. Most of these deep learning methods rely on convolutional neural networks (CNN). Many also use residual blocks or highway connections to deepen the network while avoiding vanishing gradients. Two of the main state-of-the-art methods that this paper uses as benchmarks are EfficientDeRain (EDR) [[2]](#2) and rain convolution dictionary network (RCDNet) [[3]](#3).

In EDR, the input image is processed using pixel-wise exclusive kernels, unlike normal CNNs. The kernel weights are learned using a Kernel Prediction Network (KPN). These learnable kernels can output spatial-variant kernels depending on the thickness and strength of rain streaks.

The primary innovation in RCDNet is its network architecture which encodes input images using the convolutional dictionary learning mechanism. This allows a more interpretable representation of rain in the architecture, eventually leaving each rain kernel to be fully lifted from the image.

## Model Architecture

The proposed model uses U-Net architecture [[4]](#4). This architecture starts off by taking in a 256x256 RGB input image. Then, this image is fed into an encoder of one convolutional block followed by two downsampling blocks. Then, nine deformable residual blocks are used to create the latent space of 256x64x64 size. The residual blocks utilize deformable convolution [[5]](#5) in place of normal convolution layers. This allows the model to incorporate non-local spatial information and effectively reconstruct local image distortions caused by rain effects. Following this, there are two upsampling blocks and one output layer which reconstructs the 256x256 RGB de-rained image. Skip connections between the encoder and decoder are fused using 3x3 up-convolution blocks. 

<img width="700" alt="Screen Shot 2023-03-11 at 3 13 53 PM" src="https://user-images.githubusercontent.com/61512660/235407329-86cffa15-dbda-4c58-b151-78a078ab6a15.png">

During the training process of the deraining model, 256 × 256 patches are utilized, with a mini-batch size of N = 8 for a duration of 20 epochs. The Adam optimizer is implemented with values of $β_1 = 0.9$ and $β_2 = 0.999$. The initial learning rate is set at 2 × 10<sup>-4</sup> and is gradually adjusted to 1 × 10<sup>-6</sup> according to a cosine annealing schedule. A linear warm-up policy is also implemented for the first 4 epochs. To enhance data variation, random cropping, random rotation, random horizontal and vertical flips, and RainMix augmentation [[2]](#2) are employed.
	
The loss function is constructed using three different metrics. Firstly, the de-rained image is compared to the clean image using a basic l1 image loss. 

Secondly, a multi-scale structural similarity index measure (MS-SSIM) score is calculated between the two images. MS-SSIM is a modified form of SSIM, which is described in detail later on. 

Lastly, a novel rain-robust loss is calculated. Previous methods typically de-rain an image by directly learning a mapping to the estimated output through minimizing the image reconstruction loss between the de-rained image and the clean image. However, this approach involves exploring a large hypothesis space, which is challenging because areas covered by rain streaks are often ambiguous. In contrast, the proposed rain-robust loss attempts to constrain the hypothetical learning space. This is achieved by training the encoder to map both the rainy and clean images to an embedding space where they are closely located to each other.  

Technically, this is achieved by feeding both the rainy and clean images into a shared-weight encoder before the rain-robust loss is calculated on a resulting downsampled latent vector. The loss function for a positive pair of latent vectors $z_{J_{i}}$ and $z_{I_{i}}$ is given as: 

$$ℓ_{z_{J_{i}} , z_{I_{i}}} = -log{exp(sim_{cos}(z_{I_{i}}, z_{J_{i}}) / {\tau}) \over \sum_{k∈K}exp(sim_{cos}(z_{J_{i}}, k) /{\tau}}$$

where $K = \\{z_{J_{i}}, z_{I_{i}}\\} ^{N}_ {j=1, j\ne i}$ where $K$ is the combination of all possible negative pairs and $sim_{cos}(u,v) = {u^Tv \over ∥u∥∥v∥}$ is the similarity cosine between any two vectors. {\tau} is set to 0.25.

Through this loss function, positive pairs are placed close together in the embedding space while negative pairs are spaced far apart. This loss-function is based on noise-contrastive estimation (NCE), specifically InfoNCE [[6]](#6). 

The final loss function takes the form of:

$$L_{full}(J^* , J) = L_{MS-SSIM}(J^* , J) + λ_{l1}L_{l1}(J^* , J) + λ_{robust}L_{robust}(J^* , J)$$

where $J$ represents the clean image and $J^*$ represents the de-rained image. $λ_{l1}$ and $λ_{robust}$ are hyperparameters set to 0.1.
## Dataset

The efficacy of any supervised model is highly dependent on the quality of the training data provided. Specifically, the model would require quality pairs of rainy and clean images of the same scene. However, as described earlier, it is impossible to observe the same scene at the exact same time and space both with and without rain. Due to the ill-posed nature of the rainy and clean image collection problem, all potential training datasets will be unideal for supervised training. Prior to the proposed dataset, the literature relied heavily on rain simulation to create datasets and was limited by the sim2real gap.

Synthetic rain datasets are either partially or fully generated. Image pairs are created by taking a clean image and applying a rain mask over it. Datasets like NYU-Rain [[7]](#7), RainCityscapes [[8]](#8), Outdoor-Rain [[9]](#9), and Rain100H [[10]](#10) are built off pre- existing image datasets which include depth maps. Then according to the depth map, a non-learnable algorithm applies rain streaks of various scales and orientations to the image. While extraneous factors in synthetic datasets are much easier to control, current methods are unable to replicate the complex effects of rain. This leads to generalizability issues on actual rainy images.

<img width="800" alt="Picture3" src="https://user-images.githubusercontent.com/61512660/235407429-0afab363-b1a2-479e-9fb7-da0c029b20da.png">

Explaining the photometry of raindrops is a complex task involving the refraction, specular reflection, and internal reflection of the raindrop. This relationship is modeled with the equation:

$$L(n) = L_{r}(n) + L_{s}(n) + L_{p}(n)$$

where $L(n)$ is the raindrop’s radiance with respect to a normal vector along its surface, $L_{r}(·)$ is the radiance of the refracted ray, $L_{s}(·)$ is the radiance of the specularly reflected ray, and $L_{p}(·)$ is the radiance of the internally reflected ray.

Beyond these factors, the appearance of rain streaks in real images is also influenced by motion blur and background intensities, while dense rain accumulation leads to intricate veiling effects. The interactions between these complex phenomena make it difficult to generate realistic simulations of rain effects.

Real rain datasets were introduced in GT-Rain. Instead of creating rainy images from non-rainy images, rainy and non-rainy image pairs are created by scraping online live streams to find videos that include rain. There are then five pre-processing steps involved in order to produce the final positive pair. Firstly, there is a general scene selection criterion. For example, footage shot at night or with raindrops on the lens would be filtered out. Secondly, the footage must capture a definitive stop in the rain. If the rain slowly fizzles out, for example, there would be too much of a time difference between the rainy and non-rainy versions of the image. Thirdly is image cropping. A large boat moving across the screen, for example, would cause a false negative as the loss from the boat movement would completely dominate the loss caused by rain. As such, the rainy image is captured seconds before the rain stops and the non-rainy image is captured seconds after the rain stops. Fourthly, SIFT and RANSAC methods are applied in order to eliminate small camera movements between the frames caused by factors such as wind. Lastly, when necessary, elastic image registration is performed by predicting a displacement field in order to correct for factors such as leaves moving. 

<img width="800" alt="Picture4" src="https://user-images.githubusercontent.com/61512660/235407516-6da3b738-d3b9-42fb-98cc-c46c0160263d.png">

While this most realistically models the complex nature of rain, the temporal aspect introduces variability factors between the two images. Factors such as illumination, local movement (i.e., foliage/object movement within the shot frame), or camera movement can change in between recording the rainy and non-rainy data.

<img width="300" alt="Picture5" src="https://user-images.githubusercontent.com/61512660/235407555-c301afed-463b-44e5-848d-ada509ca718d.png">

<img width="400" alt="Picture6" src="https://user-images.githubusercontent.com/61512660/235407586-1c502d94-1366-4fdf-8c7c-399fd3fc7150.png">


## Results

De-raining methods use peak signal-to-noise ratio (PSNR) [[11]](#11) and structural similarity index measure (SSIM) [[12]](#12) in order to assess performance. 

PSNR is defined by

$$PSNR ∶= 20log_{10}({MAX_f \over \sqrt{MSE}})$$

where $MAX_f$ is 255, the maximum output on a luminance (monochromatic) channel, and mean squared error $(MSE)$ between images $𝐼$ and $𝐾$ is defined as:

$$MSE := {1 \over mn} \sum^{m-1}_ {i=0}\sum^{n-1}_ {j=0}‖𝐼(𝑖, 𝑗) − 𝐾(𝑖, 𝑗)‖^2$$

where $m$ and $n$ represent the pixel width and height of $𝐼$ and $𝐾$. PSNR is represented in decibels.

SSIM is slightly more complicated, being defined as:

$$𝑆𝑆𝐼𝑀 ∶= 𝑙(𝐼, 𝐾)^\alpha * 𝑐(𝐼, 𝐾)^\beta * 𝑠(𝐼, 𝐾)^\gamma$$

$$𝑙(𝐼,𝐾):= {{2\mu_I\mu_K + c_1} \over {\mu_I^2 + \mu_K^2 + c_1}}$$

$$c(𝐼,𝐾):= {{2\sigma_I\sigma_K + c_1} \over {\sigma_I^2 + \sigma_K^2 + c_1}}$$

$$s(𝐼,𝐾):= {{\sigma_{IK} + c_3} \over {\sigma_I \sigma_K + c_3}}$$

where $\mu_I$ and $\mu_K$ are the pixel sample means of $𝐼$ and $𝐾$; $𝜎_I$ and $𝜎_K$ are the standard deviations of $𝐼$ and $𝐾$; $𝜎_{IK}$ is the covariance between $𝐼$ and $𝐾$; $𝑐_1$, $𝑐_2$, $𝑐_3$ are fixed parameters 0.0001, 0.0009, and 0.00045 respectively. SSIM works as a better approximation of human perception by putting more weight on examining the relation between spatially close pixels. $𝑙$, $𝑐$, and $𝑠$ represent the luminance, contrast, and structure of the pixels with hyper-parameter weights $𝛼$, $𝛽$, and $𝛾$.

Qualitatively, we can see that compared to other methods, the proposed model seems to be more effective at completely removing rain.

<img width="700" alt="Picture7" src="https://user-images.githubusercontent.com/61512660/235407642-c81c708b-6ee5-4d68-a80e-6780f74fa287.png">

<img width="710" alt="image" src="https://user-images.githubusercontent.com/61512660/235407648-36fffa91-860c-4047-8953-8aea06ad93bd.png">

The study conducts four separate experiments. Firstly, other methods are trained on the dataset which they work best on and are compared to the proposed model trained on GT-Rain. Using the GT-Rain test set as a performance benchmark, we can see that the proposed model outperforms all other methods. There are two observed trends in these methods: firstly, training on more synthetic data results in improved performance, as seen in MSPFN and MPRNet. Secondly, training on semi-real data, as demonstrated in SPANet, can also enhance results. However, even when using multiple synthetic or semi-real datasets, their performance on real data is still approximately 2 dB lower than training on GT-RAIN.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/61512660/235407670-7fdbf7cb-f477-483f-b876-1953387807e8.png">

The second experiment involves re-training other methods on the GT-Rain dataset. As shown, all methods demonstrated improved performance on PSNR and SSIM tests when trained on real images. 

<img width="700" alt="image" src="https://user-images.githubusercontent.com/61512660/235407722-9ce9aa51-6e30-444d-ae23-01db71a8f7c2.png">

The third experiment tests fine-tuning existing methods by further training off-the-rack methods on the GT-Rain. Essentially, the methods are first trained on a synthetic dataset and then further trained on GT-Rain. The model adopts 20% of its original learning rate when training on GT-Rain. With every method, the fine-tuned model performs better than the original model. 

<img width="700" alt="image" src="https://user-images.githubusercontent.com/61512660/235407734-858b6826-7570-4f14-9ee7-8adee60a03b9.png">

The last experiment ran was an ablation study to confirm the effectiveness of the rain-robust loss. Two versions of the proposed models, one with and one without rain-robust loss, are trained and compared to each other. The ablation study shows that the model with rain-robust loss significantly outperforms the version without. This is corroborated by examining the latent space. The normalized correlation between rainy and non-rainy latent vectors is 0.95 +/- 0.03 with the rain-robust loss whereas it is 0.85 +/- 0.10 for the model without rain-robust loss. 

<img width="700" alt="image" src="https://user-images.githubusercontent.com/61512660/235407750-3222b69f-d162-42c7-bfad-c3c06fab8caf.png">

## References

<a id="1">[1]</a> 
Kang, L., Lin, C., and Fu, Y. (2012). Automatic single-image-based rain streaks removal via
image decomposition. IEEE TIP, vol. 21, 1742-1755.

<a id="2">[2]</a> 
Guo, Q., et al. (2021). EfficientDeRain: learning pixel-wise dilation filtering for high- efficiency single-image deraining. AAAI Conference on Artificial Intelligence. vol. 35. 1487- 1495.

<a id="3">[3]</a> 
Wang, H., Xie, Q., Zhao, Q., Meng, D. (2020). A model-driven deep neural network for single image rain removal. IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3100-3109.

<a id="4">[4]</a> 
Olaf, R., Fischer, P., and Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer assisted intervention. 234–241.

<a id="5">[5]</a> 
Zhu, X., Hu, H., Lin, S., & Dai, J. (2019). Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9308-9316).

<a id="6">[6]</a> 
Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.
Chicago	

<a id="7">[7]</a> 
Li, R., Cheong, L. F., & Tan, R. T. (2019). Heavy rain image restoration: Integrating physics model and conditional adversarial learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1633-1642.

<a id="8">[8]</a> 
Hu, X., Fu, C. W., Zhu, L., & Heng, P. A. (2019). Depth-attentional Features for Single- image Rain Removal. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition. 8022-8031.

<a id="9">[9]</a> 
Li, R., Cheong, L. F., & Tan, R. T. (2019). Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1633-1642.

<a id="10">[10]</a> 
Yang, W., Tan, R. T., Feng, J., Liu, J., Guo, Z., & Yan, S. (2017). Deep Joint Rain Detection and Removal from a Single Image. Proceedings of the IEEE conference on computer vision and pattern recognition. 1357-1366.

<a id="11">[11]</a> 
Huynh-Thu, Q., & Ghanbari, M. (2008). Scope of Validity of PSNR in Image/video Quality Assessment. Electronics letters, 44(13), 800-801.

<a id="12">[12]</a> 
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE transactions on image processing, 13(4), 600-612.
