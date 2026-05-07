# VAE (Variational Autoencoder)

Auto-Encoding Variational Bayes  
KINGMA, Diederik P.; WELLING, Max. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

---

## Author's Intention

VAE is a generative model that combines variational inference with deep learning.

Let's learn a continuous and structured latent space from which new data can be generated.

Let's approximate the intractable posterior distribution using a neural network (encoder).

Let's train the encoder and decoder simultaneously end-to-end using backpropagation.

---

## Existing Problems

Traditional autoencoders compress input into a fixed latent vector and reconstruct it.

However, this latent space has no structure — sampling an arbitrary point produces meaningless output.

This makes it impossible to generate new data from the model.

The core difficulty is that the true posterior p(z|x) is **intractable** (cannot be computed directly).

p(z|x) = p(x|z) · p(z) / p(x), and p(x) = ∫ p(x|z) p(z) dz requires integrating over all possible z, which is computationally infeasible.

---

## Basic Structure of VAE

VAE consists of three components: **Encoder**, **Reparameterization**, and **Decoder**.

```
Input x  →  Encoder q_φ(z|x)  →  (μ, σ)  →  z = μ + σ·ε  →  Decoder p_θ(x|z)  →  Output x'
```

Unlike a standard autoencoder, the encoder outputs a **probability distribution** (μ, σ) rather than a single point.

The decoder then samples from this distribution to reconstruct the input.

---

## Key Concepts

### 1. Information Theory Background

Understanding VAE requires knowledge of three concepts from information theory.

**Self-Information**

I(x) = -log P(x)

The rarer an event, the higher its information content.

**Shannon Entropy**

H(X) = -Σ P(x) log P(x)

The expected value of self-information. Measures average uncertainty of a distribution.

**KL-Divergence**

D_KL(P || Q) = Σ P(x) log [P(x) / Q(x)] ≥ 0

Measures the difference between two probability distributions.

Always non-negative (proven by Jensen's Inequality: -log is convex).

Asymmetric: D_KL(P||Q) ≠ D_KL(Q||P), so it is called "divergence" not "distance".

In VAE, KL-divergence is used to force the encoder output q_φ(z|x) to be close to the standard normal prior p(z) = N(0, I).

---

### 2. Variational Inference (ELBO)

Since p(z|x) is intractable, VAE introduces a neural network q_φ(z|x) to **approximate** the true posterior.

The goal is to maximize log p_θ(x), but this is also intractable due to the integral over z.

Instead, VAE derives a tractable **lower bound** called ELBO (Evidence Lower BOund).

**Derivation (5 steps)**

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Multiply by ∫ q_φ(z\|x) dz = 1 | log p_θ(x) = ∫ q_φ · log p_θ(x) dz |
| 2 | Apply Bayes Rule | log p_θ(x) = ∫ q_φ · log [p_θ(x\|z)·p(z) / p_θ(z\|x)] dz |
| 3 | Insert q_φ(z\|x) / q_φ(z\|x) | Prepare for KL form |
| 4 | Separate log into 3 terms | Reconstruction + Regularization + intractable term |
| 5 | Use KL ≥ 0 to remove intractable term | Obtain lower bound |

**Final ELBO**

log p_θ(x) ≥ E_q[log p_θ(x|z)] − D_KL(q_φ(z|x) || p(z)) = L_ELBO

Maximizing ELBO pushes log p_θ(x) upward, so training proceeds by maximizing ELBO.

---

### 3. Reparameterization Trick

The encoder outputs μ and σ, and we need to sample z ~ N(μ, σ²).

**Problem**: Sampling is a stochastic operation — gradients cannot flow through it.

∂z/∂μ is undefined when z is sampled directly.

**Solution**: Separate the randomness into an external constant ε.

z = μ + σ · ε,   ε ~ N(0, I)

Now z is a **deterministic function** of μ and σ, so ∂z/∂μ = 1 and ∂z/∂σ = ε.

Gradients can flow from the decoder all the way back to the encoder.

**Two reasons for using this trick**

1. **(Essential) Backpropagation**: Without it, the encoder cannot be trained at all.

2. **(Additional) Diversity**: Different ε values each time produce slightly different outputs, enabling true generative behavior.

---

## Architecture Details

### Encoder q_φ(z|x)

Takes input x and outputs Gaussian distribution parameters μ and σ.

q_φ(z|x) ~ N(μ_φ(x), σ_φ(x)²I)

Implemented as a neural network with two output heads: one for μ, one for log σ².

### Reparameterization

Sample ε ~ N(0, I), then compute z = μ + σ · ε.

This step sits between encoder and decoder and is differentiable.

### Decoder p_θ(x|z)

Takes latent z and reconstructs x.

For image data (pixel values 0~1): assumes Bernoulli distribution → Binary Cross-Entropy loss.

For continuous data: assumes Gaussian distribution → MSE loss.

---

## Loss Function

Total Loss = Reconstruction Loss + KL Loss → Minimize

**Reconstruction Loss**

L_rec = E_q_φ(z|x) [-log p_θ(x|z)]

Measures how well the decoder reconstructs x from z.

Since the integral over all z is intractable, Monte Carlo approximation with L=1 sample is used in practice.

**KL Loss**

L_KL = D_KL(q_φ(z|x) || p(z)) = -½ · Σ (1 + log σ_j² − μ_j² − σ_j²)

Measures how close the encoder output is to the standard normal prior N(0, I).

When both encoder and prior are Gaussian, this has a **closed-form solution** (no integration needed).

**Why these two terms?**

Reconstruction Loss alone → the encoder can ignore structure (posterior collapse risk).

KL Loss alone → the latent space is regularized but reconstruction quality is lost.

The two terms balance generation quality and latent space structure simultaneously.

---

## AE vs VAE

| Category | Autoencoder (AE) | Variational AE (VAE) |
|----------|-------------------|----------------------|
| Purpose | Encoder — compression & reconstruction | Decoder — new data generation |
| Latent z | Deterministic vector (single point) | Probability distribution parameters (μ, σ) |
| Latent space | Arbitrary, unstructured | Structured, close to N(0, I) |
| New data generation | Impossible ✗ | Possible via z sampling ✓ |
| Training objective | Minimize reconstruction error | Maximize ELBO (Recon + KL) |
| Model type | Discriminative / Compression | Generative / Stochastic |

The key difference is the **latent space structure**.

AE's latent space has no defined structure, so sampling an arbitrary z gives meaningless output.

VAE forces the latent space to follow N(0, I) via KL Loss, enabling smooth interpolation and generation.

---

## Implementation (Keras)

```python
class VAE(keras.Model):
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Encoder: x → (z_mean, z_log_var, z)
            z_mean, z_log_var, z = self.encoder(data)

            # Decoder: z → reconstruction
            reconstruction = self.decoder(z)

            # Reconstruction Loss (Binary Cross-Entropy)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)))

            # KL Loss: closed-form -½ Σ(1 + logσ² − μ² − σ²)
            kl_loss = -0.5 * (1 + z_log_var
                              - tf.square(z_mean)
                              - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Total Loss
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
```

**Key points**:

- `z_log_var` stores log σ² (not σ directly) for numerical stability.
- KL Loss has a closed-form solution under the Gaussian assumption — no sampling required.
- `GradientTape` records operations for both encoder (φ) and decoder (θ), updating them simultaneously.
- The Reparameterization Trick enables gradient flow through the sampling step.

---

## Applications

**Image generation & editing**

Training on CelebA face dataset → generates faces that do not exist.

Linear interpolation between z₁ and z₂ in latent space → smooth transition between two images (impossible with AE).

Attribute control: manipulate latent vectors to change hair color, glasses, etc.

β-VAE: weighting KL Loss with β > 1 → each latent dimension captures an independent attribute (disentangled representation).

**Drug molecule design**

Embed molecular structures into a continuous latent space → generate molecules with similar properties.

Reference: "Automatic Chemical Design Using a Data-Driven Continuous Representation" (2018).

**Music & text generation**

MusicVAE: embed musical bars into latent space → interpolate between two melodies.

Text VAE: represent sentences as continuous vectors → style transfer, sentiment control.

**Anomaly detection**

Train on normal data only → abnormal data shows high reconstruction error.

Applied to industrial defect detection, medical imaging (MRI, CT).

Fully unsupervised — no labels required.

**Foundation for Stable Diffusion**

Latent Diffusion Model = VAE (compression) + U-Net (diffusion) + CLIP (conditioning).

Stable Diffusion compresses images into latent space using VAE, then applies diffusion there.

Understanding VAE is essential for understanding all modern latent diffusion models.

---

## Summary

VAE = Encoder(μ, σ) + Reparameterization(ε) + Decoder + ELBO Loss

| Component | Role |
|-----------|------|
| Encoder | Learn approximate posterior q_φ(z\|x) ~ N(μ, σ²) |
| Reparameterization | Enable backpropagation through sampling: z = μ + σ·ε |
| Decoder | Learn likelihood p_θ(x\|z) via MLE |
| Reconstruction Loss | Ensure generation quality |
| KL Loss | Regularize latent space to N(0, I) |
| ELBO | Tractable lower bound — circumvents intractable MLE |

The essence of VAE is to represent data as a probability distribution and sample from it to generate new data.

Without the Reparameterization Trick, backpropagation is impossible.

Without ELBO, the intractable MLE problem cannot be solved.

Without KL Loss, the latent space has no structure and generation fails.

---

[참고]

https://arxiv.org/abs/1312.6114  
https://angeloyeo.github.io/2020/10/26/information_entropy.html  
https://hyunw.kim/blog/2017/10/27/KL_divergence.html  
https://woongchan789.tistory.com/11