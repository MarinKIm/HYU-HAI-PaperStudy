# 📄 MAE (Masked Autoencoders)

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