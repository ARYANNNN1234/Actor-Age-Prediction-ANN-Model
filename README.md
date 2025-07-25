# Age Detection from Images (Deep Learning)

## Overview
This project builds and optimizes an Artificial Neural Network (ANN) to classify age groups from facial images. It demonstrates a deep dive into neural network architecture, hyperparameter tuning, and regularization techniques using TensorFlow/Keras, aiming to build a robust and accurate age prediction model.

---

## Dataset
- **Training Data:** `agedetectiontrain/Train/` (images), `agedetectiontrain/train.csv` (labels)
- **Testing Data:** `agedetectiontest/Test/` (images), `agedetectiontest/test.csv` (labels)
- Each CSV maps image IDs to age group classes.

You can download the training dataset directly:
- [agedetectiontrain.zip](https://infyspringboard.onwingspan.com/common-content-store/Shared/Shared/Public/lex_auth_012776431940165632236_shared/web-hosted/assets/agedetectiontrain.zip)
  - Place this file in your project directory and extract its contents.

---

## Methodology

### 1. Data Loading and Preprocessing
- Load image file paths and corresponding age labels from CSV files.
- Iterate through image files, load, and resize them to 32x32 pixels.
- Convert images to NumPy arrays and normalize pixel values to [0, 1].
- Encode categorical age labels using `LabelEncoder` and one-hot encode with `to_categorical`.

### 2. Neural Network Architecture
- **Model:** Sequential Keras model
    - InputLayer for 32x32x3 image data
    - Flatten layer for 2D to 1D conversion
    - Dense hidden layer (500 units, ReLU activation)
    - Dense output layer (3 units, Softmax activation for multi-class classification)

### 3. Extensive Experimentation & Optimization
- **Activation Functions:** Linear, Sigmoid, Tanh, ReLU, Softmax
- **Optimizers:** Adam, Adagrad, SGD, RMSprop
- **Learning Rate Scheduling:** Cyclic Learning Rate (CLR) with `triangular2` policy
- **Weight Initializers:** RandomNormal, GlorotNormal, HeNormal, Orthogonal
- **Bias Initializers:** Zeros, Constant(0.01), Constant(0.1)
- **Regularization Techniques:**
    - Dropout (0.3 rate)
    - MaxNorm constraint
    - L1/L2 (Elastic Net) kernel regularizers
    - EarlyStopping callback (monitoring validation loss)

### 4. Model Training and Persistence
- Compile with `categorical_crossentropy` loss and Adam optimizer (also tested alternatives).
- Train for multiple epochs (batch size: 128, validation split: 20%).
- Save final model as `optimum_model.h5`.

---

## How to Run the Project

### 1. Dataset Structure

```
your_project_folder/
├── agedetectiontrain/
│   ├── Train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── train.csv
└── agedetectiontest/
    ├── Test/
    │   ├── image_test1.jpg
    │   └── ...
    └── test.csv
├── your_script.py
└── optimum_model.h5  # After training
├── agedetectiontrain.zip  # Download and extract this file
```

### 2. Install Dependencies

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras imageio Pillow tensorflow-keras-lr-finder
```
> _Note: `tensorflow-keras-lr-finder` may require specific TensorFlow/Keras versions. If you have issues, adjust your environment or comment out the related code for basic model training._

### 3. Run the Script

```sh
python your_script.py
```

This will:
- Load and preprocess the data
- Train the optimized ANN model
- Save the trained model as `optimum_model.h5`
- Predict on test data and save results to `out.csv`
- Display a visual prediction example

---

## Files Included
- `your_script.py` - Main Python script for model definition, training, and evaluation
- `agedetectiontrain/` - Training images and `train.csv`
- `agedetectiontest/` - Testing images and `test.csv`
- `optimum_model.h5` - Trained Keras model
- `out.csv` - Predictions on the test set
- `agedetectiontrain.zip` - Downloadable training dataset

---

## Future Enhancements
- Implement Convolutional Neural Networks (CNNs) for higher accuracy
- Explore transfer learning using pre-trained models (VGG, ResNet, etc.)
- Develop a Streamlit or Flask app for interactive age prediction from uploaded images
- Integrate advanced data augmentation for improved generalization

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact
For questions or suggestions, please open an issue or contact [repo owner](https://github.com/ARYANNNN1234).
