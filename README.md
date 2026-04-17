# BirdCLEF - Bird Species Identification from Audio Data

## 📌 Overview
This project was developed to identify various bird species from audio recordings. The core approach involves converting raw 1D audio signals into 2D Mel-spectrograms and treating the task as an image classification problem using a Convolutional Neural Network (CNN). 

## 🚀 Results
<img width="1890" height="1330" alt="image" src="https://github.com/user-attachments/assets/41ec08b5-d6c1-4e81-88d5-932f0d974624" />

* **Kaggle Public Score:** 0.73

## 🛠️ Implementation Details

### 1. Data Preprocessing (`train-databirdclef.ipynb`)
Audio processing is handled primarily using `librosa`. To prepare the data for the CNN, the following steps are applied:
* **Segmenting:** Audio files are split or padded into consistent 5-second chunks.
* **Mel-Spectrogram Conversion:** The audio waves are transformed into Mel-spectrograms (128 Mel bands, FMIN=20, FMAX=14000).
* **Resizing & Normalization:** Spectrograms are converted to a decibel (dB) scale, normalized, and resized to $224 \times 224$ pixels using OpenCV to match the expected input shape for standard pre-trained vision models.
* **Caching:** Processed datasets are saved as PyTorch `.pt` tensors for faster loading during training.

### 2. Model Architecture (`efficientnet-birdclef-25.ipynb`)
* **Backbone:** Uses **EfficientNet-B3** for robust and efficient feature extraction.
* **Custom Head:** The network's classifier head is replaced with a custom sequential block featuring Global Average Pooling (GAP), linear layers, Batch Normalization, and Dropout to map the extracted features to the target bird classes.

### 3. Training Pipeline
The training pipeline is built in PyTorch and optimized to prevent overfitting on noisy environmental audio:
* **Optimizer:** **AdamW** with a learning rate of `3e-4` and weight decay (`0.01`). Differential learning rates are used (the backbone trains at 1/10th the speed of the classifier head).
* **Loss Function:** **Label Smoothing Cross-Entropy** (smoothing factor of 0.1) is applied to improve generalization and handle noisy or uncertain labels.
* **Scheduler:** Cosine Annealing Learning Rate Scheduler.
* **Augmentations:** * *Audio level:* Random time shifting and Gaussian noise injection.
    * *Spectrogram level:* **SpecAugment** (Time and Frequency masking) via `torchaudio.transforms`.

### 4. Inference & Submission (`submission.ipynb`)
The inference script is designed to run in a constrained Kaggle environment:
* Loads continuous test soundscapes.
* Iteratively slices the audio into sequential 5-second windows.
* Generates Mel-spectrograms dynamically, feeds them through the trained model, and outputs softmax probabilities for each class.
* Aggregates the predictions into the final `submission.csv` format required by the competition.

