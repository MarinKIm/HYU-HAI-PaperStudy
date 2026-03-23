# MAE (Masked Autoencoders)

Masked Autoencoders Are Scalable Vision Learners  
HE, Kaiming, et al. Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022. p. 16000-16009.

---

## Author's Intention

MAE is a scalable self-supervised model in the CV field.

Let's train large models efficiently and effectively.

Let's approach models with high capacity that generalize well.

---

## Existing Problems

In NLP, the masked autoencoder method is used successfully in the BERT model, but in CV its performance is relatively lower.

Transformer models like BERT were originally designed for NLP tasks. Therefore, they reflect language characteristics well.

However, in CV, transformers are being applied somewhat forcefully. This requires precise adaptation.

Let's think about what differences between NLP and CV cause differences in MAE.

---

## Basic Structure of MAE

<img width="460" height="284" alt="Image" src="https://github.com/user-attachments/assets/d8171274-f75b-42d7-bf54-82390795024e" />

1. Architecture difference between CNN and BERT

: CNN is based on convolution networks, and BERT is based on attention networks.

However, this issue has been somewhat resolved with the emergence of ViT.

2. Information density

3. Decoder part of Autoencoder

---

## Solution Idea

### 1. Masking a high portion

Language is a signal created by humans, while images are signals created by nature.

Language has higher information density compared to images. (Words contain grammar, meaning, context, etc.)

We must recognize that these two signals are fundamentally different.

NLP models predict only a small number of words (as in BERT's MAE),

but in CV, we should not apply only a small amount of masking.

This is because the level of information density is different.

Therefore, the authors applied a higher masking portion.

This strategy resolves information redundancy and requires understanding of the entire image.

---

### 2. Masking random patches

<img width="329" height="229" alt="Image" src="https://github.com/user-attachments/assets/7d970a11-b337-43f9-aa45-1abddef47d0b" />

Masked image / MAE reconstruction / Ground-truth

pixel reconstruction task : predicting the values (R, G, B) of unknown pixels.

recognition task : understanding an image and predicting its class.

In CV, the decoder performs pixel reconstruction, which is a low semantic level task compared to recognition.

On the other hand, in NLP, the decoder reconstructs missing words, which is a high semantic level task.

Therefore, instead of pixels, patches should be reconstructed, and tokens (=positions) should be masked to create a task of similar difficulty.

Thus, the authors randomly masked 'patches'.

---

### 3. Approach

In this section, the architecture and training details are explained.

---

### 3-1. Masking

<img width="470" height="292" alt="Image" src="https://github.com/user-attachments/assets/93fea0fe-7738-41e3-b423-cd3d9039a282" />

[Masking]

ViT divides the entire image into patches, and MAE follows the same process.

Before input, patches are randomly sampled at a high ratio and masked (removed).

Through this method, information redundancy can be reduced.

It also prevents the task from being solved too easily using neighboring patches.

---

### 3-2. MAE encoder

<img width="470" height="292" alt="Image" src="https://github.com/user-attachments/assets/68ca382f-33fa-48ed-ad69-0c9d599a2653" />

[encoder]

The MAE encoder is the same as ViT, but only the unmasked patch 'subset' is used as input.

Both input and output consist of a small portion (about 25%) of the patches from the full image.

As shown in the figure, out of 25 input patches, only 8 are fed into the encoder.

This allows training very large encoders while using only a fraction of computation and memory.

---

### 3-3. MAE decoder

<img width="470" height="292" alt="Image" src="https://github.com/user-attachments/assets/ccfa5a15-593f-4741-a4a5-cc46b7d2c991" />

[decoder]

The MAE decoder takes both unmasked and masked patches, i.e., the 'full set', as input.

As shown in the figure, the encoder output of 8 patches is expanded back to 25.

Positional embeddings are added to all patches (=tokens).

As shown in the figure, they are restored to their original positions.

The MAE decoder is used only during pre-training for the pixel reconstruction task.  
It is designed to be smaller, lighter, and shallower than the encoder (lightweight decoder).

---

### 3-4. Reconstruction target

<img width="470" height="292" alt="Image" src="https://github.com/user-attachments/assets/fd04085e-dab2-4bd9-b668-c7057607cd55" />

[target]

Predict the pixel values of each masked patch.

The decoder output reconstructs the image.

As the loss function, MSE is computed in pixel space.

Like BERT, the loss is calculated only on masked patches.

(This means unmasked patches are not used in loss calculation.)

## Results Analysis

---

### 1. ImageNet Experiments

---

#### 1-1. Baseline: ViT-L/16

<img width="438" height="60" alt="Image" src="https://github.com/user-attachments/assets/ba8b5d8a-42fb-4afe-98fc-51d37999ee14" />

**accuracy**  

---

<img width="438" height="237" alt="Image" src="https://github.com/user-attachments/assets/7f0ae396-a89f-452c-bd92-97cbf1219e55" />

**pre-training setting**  
Pre-training was performed on ImageNet-1K using a reconstruction task with self-supervised learning,

and fine-tuning was performed on ImageNet-1K using a recognition task with supervised learning.

MAE only used 50 epochs for fine-tuning. (In contrast, ViT-L/16 used 200 epochs.)

---

<img width="438" height="473" alt="Image" src="https://github.com/user-attachments/assets/6ccaa601-5c36-4766-80a3-29449f339253" />

**fine-tuning setting**  
Fine-tuning and linear probing use separate recipes.

---

<img width="438" height="304" alt="Image" src="https://github.com/user-attachments/assets/5aad44d4-a7d9-429e-82ff-e3231742d97c" />

**fine-tuning vs linear probing**  
Let’s examine the difference between fine-tuning and linear probing using MAE.

Fine-tuning trains both the pre-trained encoder and the MLP head,

while linear probing freezes the pre-trained encoder weights and trains only the MLP head.

---

#### 1-2. Masking ratio

<img width="438" height="277" alt="Image" src="https://github.com/user-attachments/assets/bded044f-c18f-4f06-8ab8-22014699c107" />

**masking ratio**  
A masking ratio of 75% showed the best performance for both fine-tuning and linear probing.

This is in strong contrast to the typical 15% used in BERT.

As explained earlier, this can be interpreted as a result of the difference in information density between language and images.

Fine-tuning and linear probing show different trends depending on the ratio.

This is because pre-training is a lower-level task based on pixel reconstruction.

---

#### 1-3. Decoder design

<img width="603" height="175" alt="Image" src="https://github.com/user-attachments/assets/c135ef9b-5001-417b-b66d-3eed98eabcb0" />

**decoder design**  
The depth and width of the decoder were not very important for fine-tuning.

This can be interpreted as the decoder being removed during testing, and only the encoder is used.

Therefore, making it lightweight is beneficial, and the authors made it as light as possible.

As the depth of the decoder becomes shallower, the encoder tends to specialize more in the reconstruction task, while recognition performance is not affected.

---

#### 1-4. Mask token

<img width="414" height="232" alt="Image" src="https://github.com/user-attachments/assets/731254f5-fea4-4a12-9608-3b698f54352f" />

**mask token**  
There is about a 3.3× difference in FLOPs depending on whether mask tokens are used in the encoder or not.

(This is because 75% is masked, which is easy to understand even with simple calculation.)

Therefore, the authors did not include mask tokens in the encoder and trained efficiently.

An interesting point is that the accuracy of linear probing improved significantly.

This part is not very clear...

---

#### 1-5. Augmentation, Schedule

<img width="447" height="232" alt="Image" src="https://github.com/user-attachments/assets/17c969b8-3ff3-45b5-a1a4-67ba3b4bdb1c" />

**augmentation**  
In the case of augmentation, accuracy slightly improved with crop and random-size, but there is no significant difference even without it.

---

<img width="447" height="270" alt="Image" src="https://github.com/user-attachments/assets/22fb88f2-0bc2-4e62-92fb-b2e8ed73f2bf" />

**schedule**  
The results in this paper are basically based on training up to 800 epochs.

However, even when training up to 1600 epochs, it does not show saturation.

---

#### 1-6. Compare with previous results

<img width="505" height="209" alt="Image" src="https://github.com/user-attachments/assets/d9199a2b-5bf6-456c-b938-45a7f2f90021" />

**compare**  
The results of MAE (fine-tuning) were compared with other self-supervised ViT models.

MAE achieved 87.8% accuracy when fine-tuned with image size 448 (best).

It slightly outperforms previous models.

This model is based only on a vanilla ViT, and more advanced models may achieve better results (e.g., BEiT, DeiT, DaViT, etc.).

---

### 2. Transfer Learning Experiments
