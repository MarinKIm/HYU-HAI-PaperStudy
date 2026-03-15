<h1>Vision Transformer (ViT) Explained</h1>

<p>
Vision Transformer is a model that processes images using the Transformer architecture.
Traditional image models relied on CNNs, but ViT divides an image into patches and feeds them into a Transformer.
</p>

<p><b>Representative Paper:</b></p>
<ul>
<li>An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale</li>
</ul>

<hr>

<h2>1. Overall Architecture</h2>

<p>The overall pipeline of Vision Transformer is as follows.</p>

<pre>
Image
 ↓
Patch Split
 ↓
Patch Embedding
 ↓
+ CLS Token
 ↓
+ Position Embedding
 ↓
Transformer Encoder (L layers)
 ↓
CLS Token Output
 ↓
MLP Classifier
 ↓
Image Classification
</pre>

<p>
The key idea is to treat an image like a sequence of words.
</p>

<img width="530" height="113" alt="Image" src="https://github.com/user-attachments/assets/ee58b9c8-e9e0-4a53-869d-9e58bfb7f8bc" />


<hr>

<h2>2. Image → Patch</h2>

<p>Input image</p>

<pre>
x ∈ R^(H × W × C)
</pre>

<ul>
<li>H : image height</li>
<li>W : image width</li>
<li>C : channel (RGB = 3)</li>
</ul>

<p>The image is divided into <b>P × P patches</b>.</p>

<p>Number of patches</p>

<pre>
N = HW / P²
</pre>

<p><b>Example</b></p>

<pre>
224 × 224 image
patch size = 16 × 16
N = 196 patches
</pre>

<hr>

<h2>3. Patch Embedding (Eq.1)</h2>

<p>
Each patch is flattened and transformed into an embedding vector through a linear projection.
</p>

<pre>
z₀ = [x_class; x_p¹E; x_p²E; ... ; x_pᴺE] + E_pos
</pre>

<p><b>Explanation</b></p>

<ul>
<li>x_class : CLS token</li>
<li>x_p^i : i-th patch</li>
<li>E : patch embedding matrix</li>
<li>E_pos : position embedding</li>
</ul>

<p><b>Process</b></p>

<pre>
patch
 ↓
flatten
 ↓
linear projection
 ↓
patch embedding
</pre>

<p>
Since the Transformer cannot understand order information by itself, position embedding is added.
</p>

<hr>

<h2>4. Transformer Encoder</h2>

<p>The Transformer encoder consists of two blocks.</p>

<p>1️⃣ Multi-head Self Attention</p>
<p>2️⃣ MLP Block</p>

<p>This structure is repeated <b>L times</b>.</p>

<hr>

<h2>5. Self-Attention Block (Eq.2)</h2>

<pre>
z'ℓ = MSA(LN(zℓ−1)) + zℓ−1
</pre>

<p><b>Explanation</b></p>

<ul>
<li>MSA : Multi-head Self Attention</li>
<li>LN : Layer Normalization</li>
<li>+ zℓ−1 : Residual connection</li>
</ul>

<p>
Self-Attention learns relationships between patches.
</p>

<p><b>Example</b></p>

<pre>
patch1 ↔ patch2
patch1 ↔ patch3
patch1 ↔ patchN
</pre>

<p>
In other words, it learns global relationships within the image.
</p>

<hr>

<h2>6. MLP Block (Eq.3)</h2>

<pre>
zℓ = MLP(LN(z'ℓ)) + z'ℓ
</pre>

<p><b>MLP Structure</b></p>

<pre>
Linear
 ↓
GELU activation
 ↓
Linear
</pre>

<p><b>Role of MLP</b></p>

<ul>
<li>Apply a non-linear transformation to features obtained from attention</li>
<li>Make the representation more expressive</li>
</ul>

<hr>

<h2>7. Image Representation (Eq.4)</h2>

<p>
The CLS token from the final Transformer layer is used as the image representation.
</p>

<pre>
y = LN(z_L^0)
</pre>

<p>Here</p>

<pre>
z_L^0
</pre>

<p>
is the output vector of the CLS token.
This vector summarizes the information of the entire image.
</p>

<hr>

<h2>8. Classification</h2>

<p><b>Final structure</b></p>

<pre>
CLS vector
 ↓
MLP classifier
 ↓
Image label
</pre>

<p><b>Example</b></p>

<pre>
cat
dog
car
</pre>

<hr>

<h2>9. Key Ideas</h2>

<p>Main concepts of Vision Transformer</p>

<ol>
<li>Divide the image into patches</li>
<li>Convert patches into embedding vectors</li>
<li>Learn relationships between patches using a Transformer encoder</li>
<li>Use the CLS token for image classification</li>
</ol>

<hr>

<h2>10. Difference from CNN</h2>

<table>
<tr>
<th>CNN</th>
<th>Vision Transformer</th>
</tr>

<tr>
<td>Convolution</td>
<td>Self-Attention</td>
</tr>

<tr>
<td>Local feature</td>
<td>Global relationship</td>
</tr>

<tr>
<td>Inductive bias strong</td>
<td>Inductive bias weak</td>
</tr>

</table>

<p><b>Key sentence</b></p>

<p>
Vision Transformer converts an image into a patch sequence and processes it using a Transformer.
</p>