# Neural Networks Introduction

## What it is

A neural network is a machine learning model inspired (loosely) by how biological neurons process information. At its core, a neural network is a series of mathematical transformations: it takes input data, passes it through one or more layers of interconnected nodes ("neurons"), applies non-linear transformations at each layer, and produces an output -- a prediction, a classification, or a generated value. The simplest architecture, a **multi-layer perceptron (MLP)**, is what we cover here: an input layer, one or more hidden layers, and an output layer, all fully connected. Neural networks learn by iteratively adjusting the strength (weights) of connections between neurons to minimize prediction error, using a process called backpropagation.

## Why it matters

Neural networks are the foundation of modern deep learning, powering everything from image recognition to natural language processing to autonomous vehicles. For an actuarial or data science practitioner, understanding neural networks matters for several reasons:

- **Tabular data modeling**: While tree-based models (XGBoost, LightGBM) generally dominate tabular data benchmarks, neural networks can be competitive or superior when you have very large datasets, complex feature interactions, or when you want to incorporate multiple data types (tabular + text + image) in a single model.
- **Embeddings for categorical variables**: Neural networks learn dense vector representations of categorical variables (entity embeddings) that can capture relationships that one-hot encoding misses.
- **Transfer learning and pre-trained models**: Understanding NNs lets you leverage pre-trained models for text (BERT), images (ResNet), and time series (temporal fusion transformers).
- **Telematics and IoT data**: Sensor data from vehicles or wearable devices produces sequence data that is naturally suited to recurrent or convolutional neural networks.
- **Career preparedness**: Neural networks are a foundational skill expected in any data science role, even if you primarily use tree-based models.

## When to use it

- You have a **very large dataset** (100K+ rows) where the additional capacity of a neural network can be leveraged without overfitting.
- Your data includes **multiple modalities**: tabular features plus text, images, or sequences. Neural networks can process all of these in a single architecture.
- You want to learn **entity embeddings** for high-cardinality categorical variables (e.g., ZIP codes, vehicle make/model codes).
- You are working with **sequence data** (time series of sensor readings, text sequences) where recurrent or convolutional architectures excel.
- Standard linear models or tree-based models have plateaued and you want to explore more complex function approximations.
- You need **differentiable models** that integrate with gradient-based optimization pipelines.

### When NOT to use neural networks

- **Small datasets** (<10K rows): Neural networks need lots of data to learn effectively. Tree-based models generalize better on small data.
- **Tabular data with moderate size**: XGBoost, LightGBM, and CatBoost consistently outperform MLPs on standard tabular benchmarks. Start with trees.
- **Interpretability is critical**: Neural networks are harder to interpret than tree-based models or GLMs. If regulators or stakeholders need to understand every decision (common in insurance pricing), a neural network may not be appropriate -- or you will need additional interpretability tools (SHAP, LIME).
- **Fast prototyping**: Tree-based models require less tuning and train faster. Start simple.

## How it works (intuitive)

### The building blocks

**Neuron (node)**: A neuron takes multiple inputs, multiplies each by a weight, sums them up, adds a bias term, and passes the result through an activation function. Think of it as a tiny decision-maker that receives evidence from multiple sources, weighs the evidence, and produces a signal.

**Analogy**: A neuron is like a judge scoring a gymnast. The judge sees multiple criteria (difficulty, execution, artistry), assigns importance (weights) to each, sums them up, adjusts for personal bias, and then maps the total to a final score through their own scoring rubric (activation function).

**Layer**: A collection of neurons that all receive the same inputs (from the previous layer) and produce outputs that feed into the next layer.

**Network**: Stack layers together -- input layer, hidden layers, output layer -- and you have a neural network.

### Forward propagation (making predictions)

Think of a neural network as an assembly line:

1. **Raw materials enter** (input layer): Your features -- age, claim history, vehicle type, etc.
2. **Workstation 1** (hidden layer 1): Each worker (neuron) combines the raw materials in a specific way, creating intermediate representations. One neuron might learn to detect "young driver with sports car" patterns; another might pick up "rural area with low claim frequency."
3. **Workstation 2** (hidden layer 2): These workers take the intermediate outputs from workstation 1 and create higher-level representations -- combinations of combinations. Maybe "high-risk profile combining young age, sports car, and urban location."
4. **Final output** (output layer): Produces the prediction -- a claim probability, a severity estimate, or a risk class.

Mathematically, for a single hidden layer:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

Where $\mathbf{x}$ is the input, $\mathbf{W}_1, \mathbf{W}_2$ are weight matrices, $\mathbf{b}_1, \mathbf{b}_2$ are bias vectors, and $\sigma$ is an activation function.

### Activation functions (why non-linearity matters)

Without activation functions, a neural network is just a sequence of linear transformations, which collapses to a single linear transformation. That is just linear regression with extra steps. Activation functions introduce non-linearity, allowing the network to learn curved decision boundaries.

| Function | Formula | Range | Use case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Default for hidden layers. Fast, simple, effective |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary classification output layer |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers (centered output, but slower than ReLU) |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sums to 1 | Multi-class classification output layer |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | Avoids "dying ReLU" problem |

**Analogy**: Activation functions are like volume knobs with different response curves. ReLU is a hard cutoff -- below zero, silence; above zero, pass the signal through unchanged. Sigmoid is a smooth S-curve that compresses everything into a 0-1 range.

### Backpropagation (learning from mistakes)

This is how the network learns. The process:

1. **Forward pass**: Run input through the network and get a prediction.
2. **Compute the loss**: Compare the prediction to the true value using a loss function (e.g., mean squared error for regression, cross-entropy for classification).
3. **Backward pass**: Using the chain rule of calculus, compute how much each weight contributed to the error. This gives you the gradient -- the direction and magnitude to adjust each weight.
4. **Update weights**: Nudge each weight in the direction that reduces the error, scaled by a learning rate.
5. **Repeat** for many examples (epochs).

**Analogy**: Imagine a Rube Goldberg machine that produces the wrong result. You trace backward through the machine: "This lever was pulled too hard, which pushed this ball too far, which turned this wheel too much." You adjust each component proportionally to its contribution to the error. That is backpropagation.

### Loss functions

| Task | Loss function | Formula |
|------|---------------|---------|
| Regression | Mean Squared Error | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| Binary classification | Binary Cross-Entropy | $-\frac{1}{n}\sum[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ |
| Multi-class | Categorical Cross-Entropy | $-\sum_c y_c \log(\hat{y}_c)$ |
| Regression (robust) | Huber Loss | MSE for small errors, MAE for large errors |

### Optimization (gradient descent variants)

**Stochastic Gradient Descent (SGD)**: Update weights after each mini-batch of data. Noisy but fast.

**Adam** (Adaptive Moment Estimation): The most popular optimizer. It adapts the learning rate for each parameter based on the history of gradients. Think of it as SGD with momentum and per-parameter learning rates. Use this as your default.

**Learning rate**: The step size for weight updates. Too high: overshoots the minimum and diverges. Too low: converges painfully slowly. Common strategies: start at 0.001 with Adam, use a learning rate scheduler to decrease it over time.

### Regularization (preventing overfitting)

Neural networks have many parameters and overfit easily. Key regularization techniques:

- **Dropout**: During training, randomly set a fraction (e.g., 20-50%) of neuron outputs to zero. This forces the network to not rely on any single neuron and creates an ensemble-like effect. At test time, all neurons are active.
- **Weight decay (L2 regularization)**: Penalize large weights in the loss function.
- **Early stopping**: Monitor validation loss during training. Stop when it starts increasing (the model is beginning to overfit).
- **Batch normalization**: Normalize the inputs to each layer. Stabilizes and accelerates training.

## Key concepts

### Architecture decisions

| Hyperparameter | Typical starting point | Guidance |
|----------------|----------------------|----------|
| Number of hidden layers | 1-3 for tabular data | More layers = more capacity but harder to train |
| Neurons per layer | 64-256 | Start wide, reduce if overfitting |
| Activation (hidden) | ReLU | Change only if you have specific reasons |
| Activation (output) | Sigmoid (binary), Softmax (multi-class), Linear (regression) | Match to your task |
| Optimizer | Adam | Start here, try SGD + momentum for fine-tuning |
| Learning rate | 0.001 | Use a scheduler or reduce on plateau |
| Batch size | 32-256 | Larger = faster but less noisy gradients |
| Epochs | 50-200 with early stopping | Let early stopping decide |
| Dropout rate | 0.2-0.5 | Higher for more regularization |

### Universal Approximation Theorem

A neural network with a single hidden layer containing enough neurons can approximate any continuous function to arbitrary precision. This is a theoretical result -- in practice, deeper networks with fewer neurons per layer are more efficient learners than a single extremely wide layer.

### Neural networks vs. tree-based models for tabular data

This is an important practical consideration. Research (Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning on tabular data?") shows:

**Trees win when:**
- Dataset is small to moderate (<100K rows)
- Features include irregular, non-smooth patterns
- There are uninformative features (trees naturally ignore them)
- You need fast iteration and minimal tuning
- Interpretability is important

**Neural networks win when:**
- Dataset is very large (1M+ rows)
- You are combining tabular data with other modalities (text, images, sequences)
- You want entity embeddings for high-cardinality categoricals
- You need a differentiable pipeline (e.g., for end-to-end optimization)

**Practical advice**: For most tabular actuarial/insurance problems, start with XGBoost or LightGBM. Try a neural network if you have abundant data and complex feature interactions, or if you want to incorporate non-tabular inputs.

## Python implementation

### Basic MLP with Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Train with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
```

### MLP with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

### Entity embeddings for categorical variables

```python
# Using Keras functional API for mixed input
num_input = keras.Input(shape=(n_numeric_features,), name='numeric')
cat_input = keras.Input(shape=(1,), name='zipcode', dtype='int32')

# Embedding layer for ZIP code (10,000 unique values -> 50-dim embedding)
embedding = layers.Embedding(input_dim=10000, output_dim=50)(cat_input)
embedding = layers.Flatten()(embedding)

# Combine
combined = layers.Concatenate()([num_input, embedding])
x = layers.Dense(128, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=[num_input, cat_input], outputs=output)
```

## Gotchas & lessons learned

### Feature scaling is mandatory
Unlike tree-based models, neural networks are highly sensitive to feature scales. Always standardize or normalize your input features (StandardScaler, MinMaxScaler). Failing to do so means features with large ranges dominate the gradient, leading to slow convergence or poor performance.

### Start simple, add complexity gradually
Begin with a single hidden layer of 64-128 neurons. Only add layers/neurons if the model underfits. Deeper is not always better, especially for tabular data.

### Monitor validation loss, not training loss
The training loss will keep going down. What matters is the validation loss. Use early stopping based on validation loss to prevent overfitting.

### Learning rate is the most important hyperparameter
If your model is not learning (loss stays flat), try increasing the learning rate. If the loss is oscillating wildly, decrease it. Learning rate schedulers (ReduceLROnPlateau, CosineAnnealing) can automate this.

### Dying ReLU problem
If many of your neurons output exactly zero for all inputs, they are "dead" and will never learn again (because the gradient of ReLU is zero for negative inputs). Use Leaky ReLU or He initialization to mitigate this.

### Batch normalization helps a lot
It stabilizes training, allows higher learning rates, and provides mild regularization. Add it after each hidden layer (before or after activation -- both conventions exist).

### Do not default to neural networks for tabular data
This bears repeating: XGBoost and LightGBM are typically faster to develop, easier to tune, more interpretable, and often more accurate on tabular insurance/actuarial data. Use neural networks when you have a specific reason to do so.

### GPU acceleration
Training neural networks on CPU is slow for anything beyond small datasets. Use GPU if available (TensorFlow and PyTorch both support CUDA). Google Colab provides free GPU access for experimentation.

### Reproducibility is tricky
Set random seeds for Python, NumPy, TensorFlow/PyTorch, and CUDA. Even then, GPU operations may not be perfectly deterministic. Document your environment and seeds.

## Resources

- [Keras documentation](https://keras.io/) -- High-level API for building neural networks (TensorFlow backend)
- [PyTorch tutorials](https://pytorch.org/tutorials/) -- More flexible, research-oriented framework
- [3Blue1Brown - Neural Networks (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) -- Best visual introduction to neural networks
- [Goodfellow, Bengio, Courville (2016) - Deep Learning](https://www.deeplearningbook.org/) -- The comprehensive textbook (free online)
- [FastAI course](https://course.fast.ai/) -- Practical deep learning course, top-down approach
- [Grinsztajn et al. (2022) - Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) -- Important benchmark paper
- [Chollet (2021) - Deep Learning with Python, 2nd Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) -- Excellent practical guide by the creator of Keras
- [Ruder (2016) - An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) -- Comprehensive review of optimizers
