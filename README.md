# 🍎 Product Freshness Detection 

An automated Computer Vision prototype designed to classify fresh and anomalous (rotten) agricultural produce. This project explores real-world applications of deep learning to streamline quality control, identify anomalous items, and reduce food waste.

## 📊 Dataset
This project utilizes a preprocessed version of the "Fresh and Rotten Fruits Dataset." 
- **Total Classes:** 14 (7 fresh categories, 7 rotten categories including apples, bananas, oranges, tomatoes, potatoes, cucumbers, and okra).
- **Data Processing:** The dataset was manually balanced to ensure symmetrical classes between the training and testing environments.
- **Dataset Access:** Due to file size constraints and version control best practices, the dataset is not hosted directly in this repository. The raw images can be downloaded from the original source on Kaggle (see reference below).

### 📚 Dataset Reference
The images used to train and evaluate this model were sourced from the publicly available **Fresh and Rotten Fruits Dataset**. 
* **Platform:** Kaggle
* **Dataset Link:** `[Insert actual Kaggle link here]`
* **Modifications:** The dataset was locally modified to remove unaligned classes (capsicum, bittergourd) and correct folder naming conventions for seamless multi-class training.

**Citation:**
> [Dataset Creator's Name]. (Year). *Fresh and Rotten Fruits Dataset* [Data set]. Kaggle. `[Insert actual Kaggle link here]`

## 🧠 Methodology & Architecture
Due to the multi-class nature of the problem, this project utilizes **Transfer Learning** rather than training a CNN from scratch. 
- **Base Model:** `MobileNetV2` (Pre-trained on ImageNet).
- **Framework:** TensorFlow / Keras.
- **Modifications:** The original 1000-class classification head was removed. A Global Average Pooling layer, a Dropout layer (0.2), and a custom 14-class Dense output layer with softmax activation were added.
- **Optimization:** Adam Optimizer (learning rate = 0.0001) with Sparse Categorical Crossentropy loss. Early stopping and model checkpointing were utilized to prevent overfitting.

### Tackling Dataset Bias & Feature Confusion
During evaluation, the baseline model exhibited feature confusion when presented with out-of-distribution real-world images (e.g., classifying a dark-crimson fresh apple as "rotten" due to heavy shadowing and water droplets mimicking bruised textures). 

To solve this dataset bias, **Advanced Data Augmentation** was implemented directly into the training pipeline. By applying random adjustments to brightness and contrast (`tf.keras.layers.RandomBrightness`, `tf.keras.layers.RandomContrast`), the model was forced to stop relying on idealized studio lighting and instead learn the true structural features of the produce.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mntpps/Product-Freshness-Detection.git](https://github.com/mntpps/Product-Freshness-Detection.git)
   cd Product-Freshness-Detection

2. **Set up the environment:**
Create a new Python environment and install the required packages. Using Conda is highly recommended for managing TensorFlow dependencies on Windows:

```bash
conda create -n freshness_pj python=3.10
conda activate freshness_pj
pip install -r requirements.txt

3. **Run a Prediction:**
To evaluate a single image, use the predict.py script and pass the path to your target image:

```bash
python predict.py path/to/your/test_image.jpg
