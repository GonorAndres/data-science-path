# Computer Vision Foundations

## Image representation

An image is a 3D tensor: height x width x channels. For RGB, channels = 3; for grayscale, 1; for medical scans or satellite, can be tens or hundreds.

Pixel values are typically 0 to 255 (uint8) or normalized to 0 to 1 or mean-zero standardized. Normalization must match the preprocessing used during training (especially for pretrained models).

## Convolutions

A convolution slides a small filter over the image, computing a weighted sum at each position. Key properties:

- **Local.** Each output depends on a small neighborhood.
- **Weight sharing.** Same filter applied everywhere; translational equivariance.
- **Parameter-efficient.** Unlike fully connected layers, parameters scale with kernel size, not image size.

A convolutional layer stacks many filters in parallel, each learning a different pattern (edges, textures, parts, objects as depth increases).

### Receptive field

The region of the input influencing a given output unit. Grows with depth, stride, and dilation. Matters when deciding whether the network can see enough context for the task.

### Stride and padding

Stride: step size of the filter. Stride > 1 downsamples. Padding preserves or reduces the output size.

### Pooling

Max pool or average pool downsamples while introducing translation invariance. Modern architectures use strided convolutions instead of explicit pooling in many places.

## Classic CNN architectures

| Architecture | Year | Key idea | Parameters |
|--------------|------|----------|------------|
| LeNet-5 | 1998 | First successful CNN | ~60k |
| AlexNet | 2012 | ReLU, dropout, GPU | ~60M |
| VGG | 2014 | Deep uniform 3x3 convs | ~138M |
| GoogLeNet / Inception | 2014 | Inception modules, 1x1 convs | ~7M |
| ResNet | 2015 | Skip connections, very deep | ~25M (R-50) |
| DenseNet | 2016 | Dense concatenation | ~8M |
| EfficientNet | 2019 | Compound scaling | ~5-66M |
| ConvNeXt | 2022 | Modernized CNN matching transformers | ~28-200M |

ResNet is the default baseline in 2026 for many applications; ConvNeXt matches or beats ViTs in accuracy-per-parameter for many tasks.

## Vision Transformers (ViT)

Patch the image into 16x16 pixel blocks, flatten, linearly project each into a token. Feed the token sequence to a transformer encoder. Classification token attends over all patches.

Pros: scales with data; strong when pretrained on hundreds of millions to billions of images.
Cons: data-hungry; CNNs outperform on small datasets.

Hybrids (Swin Transformer, ConvNeXt) blend ideas from both.

## Transfer learning

The dominant pattern for practical computer vision.

1. Take a model pretrained on ImageNet (or larger: JFT, LAION).
2. Replace the final classifier head with one matching your task.
3. Fine-tune, usually with:
   - Smaller learning rate for the backbone (or freeze it for the first epochs).
   - Standard learning rate for the new head.

Typical workflow (in the portfolio's `cv-defect-detection`):

- Use `timm` to load a pretrained backbone (`resnet50`, `efficientnet_b0`, `convnext_tiny`).
- Replace final layer with a linear head of the right output size.
- Train with class-weighted cross-entropy.
- Apply data augmentation.
- Monitor per-class metrics.

## Data augmentation

Essential when data is limited.

- **Geometric.** Random crop, resize, flip, rotation, affine.
- **Photometric.** Brightness, contrast, hue, saturation jitter.
- **Mixing.** Mixup (linear combination of two images and labels), CutMix (swap a patch), CutOut (zero a patch).
- **Normalization.** Match the pretrained model's normalization stats.

Libraries: torchvision.transforms, albumentations (faster, more transforms).

Caveat: some augmentations are task-inappropriate. Flipping left-right is fine for natural images; for reading text in images, it is destructive. Know your task.

## Loss functions

- **Cross-entropy** for multi-class classification.
- **Binary cross-entropy with logits** for multi-label.
- **Focal loss** for severe class imbalance (hard negative mining).
- **Dice loss / IoU loss** for segmentation.
- **L1 / Smooth L1** for regression (bounding box coordinates).
- **Triplet / contrastive** for embedding learning.

## Common tasks

### Image classification

One label per image. Metric: accuracy, F1 (per class, macro), confusion matrix.

### Object detection

Bounding boxes + class labels per image. Architectures: Faster R-CNN, YOLOv5 through YOLOv10, DETR. Metric: mAP at IoU thresholds.

### Semantic segmentation

Pixel-level class label. Architectures: U-Net, DeepLab, Mask2Former. Metric: mean IoU.

### Instance segmentation

Pixel-level label per instance. Mask R-CNN, Mask2Former.

### Metric learning

Learn embeddings where distance corresponds to similarity. Used in face recognition, retrieval. Loss: triplet, contrastive, ArcFace.

### Self-supervised pretraining

Train on unlabeled images via pretext tasks (contrastive, masked autoencoding). Modern examples: SimCLR, MoCo, DINO, MAE. Produces embeddings useful for downstream tasks with limited labels.

## Vision-language models (VLMs)

The dominant CV paradigm in 2026 for zero-shot tasks is contrastive image-text pretraining.

### CLIP, SigLIP, BLIP

**CLIP** (Radford et al. 2021) trains an image encoder and a text encoder jointly on hundreds of millions of image-caption pairs. The objective is contrastive: for a batch of $N$ image-text pairs, score the $N \times N$ similarity matrix and maximize the diagonal while minimizing off-diagonal entries (symmetric InfoNCE). After training, both encoders project images and text into a shared embedding space.

Three downstream patterns:

- **Zero-shot classification.** Given a new task with class names $\{c_1, \ldots, c_K\}$, encode each prompt "a photo of a $c_k$" and the input image; the predicted class is the highest cosine similarity. No training data required for the new task.
- **Image retrieval.** Encode a corpus of images and a text query; return the top-K images by cosine similarity. Industrial product search, content moderation, e-commerce.
- **Item features for recommendations.** CLIP image embeddings serve as cold-start item features in two-tower recommenders. The recommender learns a mapping from (CLIP image embedding, CLIP description embedding) to a user-item scoring head; new items get useful embeddings on day zero with no interaction history.

**SigLIP** (Zhai et al. 2023) replaces the softmax in the contrastive loss with a per-pair sigmoid, removing the need for a global softmax over the batch. Trains stably with smaller batch sizes; the 2026 default for fine-tuning a CLIP-style model on a custom corpus.

**BLIP / BLIP-2** add a captioning decoder so the same model can both score (image-text alignment) and generate (image-to-caption). BLIP-2 freezes a pretrained image encoder and a pretrained LLM and trains only a small Q-Former bridge in between, making vision-language scaling cheap.

### LLaVA-style VLMs

LLaVA (Liu et al. 2023) introduced the now-standard architecture for visual instruction following:

```
Image -> [Vision Encoder, e.g. CLIP-ViT] -> visual tokens
visual tokens -> [Linear projection or MLP] -> token embeddings in LLM space
[token embeddings, text instruction tokens] -> [LLM, e.g. Llama] -> response
```

The vision encoder and the LLM are typically frozen or lightly tuned; the projection layer and a small LoRA on the LLM are the only newly trained parameters. Training data is multimodal instruction-following pairs (image + question + answer).

Use cases: visual question answering, document understanding (invoice parsing, claim form OCR-plus-reasoning), accessibility (alt-text generation), robotics (caption-conditioned action policies). For an actuarial use case: extract structured fields from photographs of property damage attached to a claim, then route to a downstream pricing model.

## Diffusion models primer

Diffusion is the dominant generative paradigm for images.

### Forward and reverse processes

The **forward process** gradually corrupts a clean image $x_0$ into pure noise $x_T$ over $T$ steps by adding Gaussian noise:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)
$$

The **reverse process** learns to invert each step. A neural network $\epsilon_\theta(x_t, t)$ predicts the noise that was added; subtracting the predicted noise gives a denoised estimate of $x_{t-1}$.

Training reduces to a simple regression: sample a clean image and a random timestep $t$, add the corresponding noise, ask the network to predict it. The connection to **score matching** is that $\epsilon_\theta$ is, up to a known scaling, an estimator of the score $\nabla_x \log p_t(x)$ of the noisy distribution at level $t$.

### DDIM and faster sampling

The original DDPM (Ho et al. 2020) requires hundreds to thousands of denoising steps to generate one image. **DDIM** (Song et al. 2021) reformulates the reverse process as a deterministic ODE; you can take much longer steps, generating a high-quality image in 20 to 50 steps. Subsequent samplers (DPM-Solver, Heun, Euler-A) push this lower; the 2026 production default is around 20 steps.

### Stable Diffusion architecture

Stable Diffusion (Rombach et al. 2022) operates in latent space:

- **VAE.** Encode the image into a compact latent representation (e.g., 64x64x4 for a 512x512 image).
- **UNet.** The denoising network operates on the latent. Much cheaper than denoising in pixel space.
- **CLIP text encoder.** Text prompt is embedded and injected into the UNet via cross-attention at every block. This is how the prompt conditions generation.

Knowing this much architecture is enough to read the literature, choose checkpoints, and decide where to fine-tune (LoRA on the UNet, textual inversion on the embedding space, ControlNet for spatial conditioning).

## Augmentation policies

Hand-designed augmentation pipelines are usually suboptimal. Three policy-search approaches:

- **AutoAugment** (Cubuk et al. 2018) uses reinforcement learning to search a discrete space of augmentation operations and probabilities. Finds policies tuned to a specific dataset.
- **RandAugment** (Cubuk et al. 2020) drops the search: at each step pick $N$ random augmentation operations from a fixed list, each applied with a single global magnitude $M$. Two hyperparameters, no search. Matches AutoAugment in practice.
- **TrivialAugment** (Muller, Hutter 2021) drops the magnitude as well: at each step pick one random operation and a random magnitude. Zero hyperparameters. Matches or beats RandAugment on standard benchmarks.

For a new project in 2026 the default is TrivialAugment: zero tuning cost, near-best results, no risk of overfitting the augmentation policy to a validation set.

## Test-time augmentation (TTA)

At inference, run the model on several augmented versions of the input image (horizontal flip, crops at different scales, color jitter), then average the per-class probabilities. The averaged prediction is more accurate and better calibrated than the single-pass prediction.

Implementation:

```python
import torch

def tta_predict(model, image, transforms_list):
    probs = []
    for t in transforms_list:
        x = t(image).unsqueeze(0)
        with torch.no_grad():
            probs.append(model(x).softmax(dim=-1))
    return torch.stack(probs).mean(dim=0)
```

When it helps: classification on noisy or low-resolution images, medical imaging, fine-grained recognition. When it does not: latency-bound serving (linear cost increase per augmentation), and tasks where augmentations break the spatial structure (detection, OCR).

## Video models

Video tasks require temporal modeling on top of spatial CNN/ViT features.

- **TimeSformer** (Bertasius et al. 2021) extends ViT with separate spatial and temporal attention. Standard supervised baseline for action recognition.
- **VideoMAE** (Tong et al. 2022) is the masked-autoencoder pretraining recipe for video; mask out tube patches across both space and time, reconstruct. Produces strong self-supervised video embeddings.

Use cases: action recognition for sports, surveillance, claim-video triage; anomaly detection in CCTV; quality control on production lines from video feeds.

## Interpretability for CNNs

- **Grad-CAM.** Uses gradients flowing into the last conv layer to produce a class-activation heatmap. Shows where the model looked.
- **Saliency maps.** Gradient of output with respect to input pixels. Pixel-level attribution.
- **Integrated gradients.** Path integral of gradients; more stable than raw saliency.
- **Occlusion.** Slide a masked window across the image, measure drop in confidence. Slow but model-agnostic.

Grad-CAM is the practical go-to for CNN debugging.

## Evaluation checklist

Beyond aggregate accuracy:

- Per-class performance. Minority classes almost always underperform.
- Confusion matrix. Which classes confuse which.
- Calibration. Is a 0.9 confidence really 90 percent correct?
- Hard examples. Inspect the 5 to 20 worst misclassifications; often reveals label noise or distribution issues.
- Fairness. Stratify by any demographic or technical attribute present.

## Production considerations

- **Inference speed.** A large ResNet can be 30 ms on GPU, 300 ms on CPU. Mobile needs optimized models (MobileNet, EfficientNet-Lite).
- **Quantization.** Int8 inference is ~4x smaller, ~2-4x faster on CPU.
- **Batching.** Large batches at inference are efficient on GPU but add latency per request.
- **Preprocessing.** Must exactly match training; forgetting normalization is a classic production bug.

## Common mistakes

1. Deploying a model trained on centered, well-lit stock images to real-world messy images. Domain shift kills performance.
2. Ignoring class imbalance. Without weighted loss or resampling, rare classes go unlearned.
3. Not freezing batch norm during fine-tuning. BN's running statistics get disrupted; use `model.train()` with care or set BN layers to eval mode.
4. Comparing models using top-1 accuracy when precision-recall on a rare class is the business metric.
5. Forgetting to check for label noise. On real datasets, 1 to 10 percent noisy labels are typical. Large gains from cleanup.

## Linking to the portfolio

- `cv-defect-detection`: end-to-end transfer learning on a defect-detection dataset (MVTec AD).
- `technique/convolutional-neural-networks`: theoretical companion with worked code.
- `technique/transfer-learning`: the mechanics of transfer learning applied in `cv-defect-detection`.

## References

- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet), 2012.
- He et al., "Deep Residual Learning for Image Recognition" (ResNet), 2015.
- Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", 2019.
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), 2020.
- Liu et al., "A ConvNet for the 2020s" (ConvNeXt), 2022.
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", 2017.
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE), 2021.

## Exercises

1. Derive the output feature-map spatial size for a convolutional layer with input $W = 224$, kernel $K = 3$, padding $P = 1$, stride $S = 2$.
2. You have 500 labeled images of industrial defects. Should you train from scratch or use transfer learning? Which backbone, and why?
3. Explain how CLIP enables zero-shot image classification without any task-specific training.
4. A diffusion model generates an image by reversing a noisy process. What does "score matching" mean in this context?
5. Compare ViT and ConvNeXt for a task with 10,000 training images. Which do you prefer, and why?
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021.
- Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP), 2023.
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", 2023.
- Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023.
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models" (DDPM), NeurIPS 2020.
- Song et al., "Denoising Diffusion Implicit Models" (DDIM), ICLR 2021.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion), CVPR 2022.
- Cubuk et al., "RandAugment", 2020.
- Muller, Hutter, "TrivialAugment", ICCV 2021.
- Tong et al., "VideoMAE", NeurIPS 2022.
