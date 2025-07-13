# 🥔 Potato Disease Classification Model

This project is an AI-powered image classification model designed to detect **potato leaf diseases** using **TensorFlow** and **Convolutional Neural Networks (CNN)**. The model can accurately classify potato plant images into one of three classes:

- `Potato___Early_blight`  
- `Potato___Late_blight`  
- `Potato___healthy`

---

## 📊 Performance Overview

Using **TensorFlow and CNN**, the model achieved:

- **Training Accuracy:** 99.80%  
- **Validation Accuracy:** 97.92%  
- **Testing Accuracy:** 96.11%

When using **ImageDataGenerator** (with basic augmentation and without custom preprocessing), the model achieved:

- **Training Accuracy:** 84.38%  
- **Validation Accuracy:** 92.71%  
- **Testing Accuracy:** 93.84%

---

## 🧠 Model Architecture

The model is built using a **Sequential CNN** architecture consisting of:

- **6 Convolutional layers** with increasing filter depths  
- **MaxPooling layers** after each convolution  
- **Flatten and Dense layers** at the end for classification  
- Final **softmax layer** for multi-class prediction  

**Total Parameters:** 183,747  
**Epochs Trained:** 50  
**Input Shape:** (32, 256, 256, 3)  
**Filter Size:** (3, 3)  
**Pooling Size:** (2, 2)

---

## 🧪 Dataset

We used the [PlantVillage dataset](https://www.kaggle.com/arjuntejaswi/plant-village) which contains categorized images of potato leaves across the three mentioned classes.

- **Training Size:** 80%  
- **Validation Size:** 10%  
- **Testing Size:** 20%  

**Dataset Directory:**  
`../../PlantVillage`

**Prepared Dataset Directory:**  
`../../dataset`

---

## 🧹 Preprocessing & Augmentation

### Custom Preprocessing:
- **Resizing**
- **Rescaling**

### Augmentation Techniques:
- **Random Flip** (horizontal & vertical)  
- **Random Rotation** (0.2)

### ImageDataGenerator Configuration:
When using only `ImageDataGenerator`, we applied:

- `rescale=1./255`
- `horizontal_flip=True`
- `vertical_flip=True`
- `rotation_range=10`
- `width_shift_range=67.`
- `height_shift_range=67.`

---

## 📚 Learning Resources

This project was inspired by and built following the amazing YouTube playlist:

📺 [Complete CNN with TensorFlow for Beginners](https://www.youtube.com/playlist?list=PLeo1K3hjS3utJFNGyBpIvjWgSDY0eOE8S)  
© All credit to [codebasics](https://www.youtube.com/c/codebasics)

---

## 🚀 How to Use

1. **Download the dataset**  
   Get it from [Kaggle - PlantVillage](https://www.kaggle.com/arjuntejaswi/plant-village)

2. **Train the model**  
   Open and run either of the following notebooks:
   - `training/training.ipynb`
   - `training_with_imageGenerator.ipynb`

3. **Run the server**  
   Follow the instructions in this repo:  
   🔗 [Potato Disease Classification Server](https://github.com/EN-BAAK/Potato-disease-classificatoin-server)

4. **Use the frontend app**  
   Follow the setup steps in:  
   🔗 [Potato Disease Classification Frontend](https://github.com/EN-BAAK/Potato-disease-classificatoin-frontend)

---

## 📦 Project Configuration

| Parameter              | Value        |
|------------------------|--------------|
| `IMAGE_SIZE`           | 256          |
| `BATCH_SIZE`           | 32           |
| `CHANNELS`             | 3            |
| `EPOCHS`               | 50           |
| `FILTERS_NUMBER`       | 32           |
| `FILTER_SIZE`          | (3, 3)       |
| `POOLING_SIZE`         | (2, 2)       |
| `SEED`                 | 200          |
| `SHUFFLE_SIZE`         | 1000         |
| `TRAINING_SIZE`        | 0.8          |

---

## 🛡 License

This project is for educational purposes and inspired by publicly available resources. Please credit the original creators and follow licensing terms from datasets and referenced materials.
