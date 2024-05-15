# Brain Tumor Segmentation Using Deep Learning

## Introduction

### Problem Definition
Brain tumor segmentation is a critical task in the field of medical imaging. It involves identifying and delineating tumor regions within brain Magnetic Resonance Imaging (MRI) scans. Brain tumors vary significantly in size, shape, location, and intensity, making them difficult to distinguish from healthy brain tissue. Furthermore, MRI scans often contain noise and artifacts, complicating the segmentation process. 

The objective of brain tumor segmentation is to develop a model that can accurately identify and segment tumor regions within MRI scans using different deep learning architectures like 3D U-Net and Inception U-Net. The ultimate goal is to create a tool that can assist radiologists and clinicians, reducing the time and effort required for manual segmentation and potentially improving patient outcomes.

## Dataset Description

The dataset utilized for this brain tumor segmentation task comprises two distinct folders (images and masks), each containing 3064 PNG images. The images are grayscale with dimensions of 512x512 pixels, depicting MRI scans of the brain.

- **Images**: Grayscale MRI scans of the brain.
- **Masks**: Binary images indicating tumor regions (white) and background (black).

Dataset: [Kaggle - Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

## Literature Review

### 1. Inception Modules Enhance Brain Tumor Segmentation

This study introduced a novel framework combining the U-Net architecture with Inception modules. Four model variations were created focusing on different combinations of glioma sub-regions and intra-tumoral structures. The integration of Inception modules improved the tumor segmentation performance significantly due to their ability to capture multi-scale contextual information efficiently.

### 2. MRI Brain Tumor Segmentation Using U-Net

The study developed an automated system using the U-Net algorithm to segment brain tumor images with high accuracy and minimal error. The system achieved a Dice Coefficient of 0.924446 and an Intersection over Union (IoU) of 0.862625, demonstrating successful tumor localization.

## Methodology

### Architectures Implemented

1. **U-Net**: Known for its success in biomedical image segmentation, the U-Net architecture features an encoder-decoder structure. The encoder extracts features from input images, while the decoder reconstructs spatial information to generate segmentation masks.

2. **Inception U-Net**: This architecture combines the robust feature extraction capabilities of Inception modules with the U-Net's segmentation prowess. Inception blocks, featuring multiple filter sizes concatenated in parallel, are integrated within the U-Net's encoder and decoder pathways to capture multi-scale information efficiently.

### Methodological Steps

1. **Data Preprocessing**:
    - Load images and masks.
    - Resize images to a standard size (256x256 pixels).
    - Normalize images.
    - Split dataset into training, validation, and test sets.

2. **Model Implementation and Compilation**:
    - Utilize TensorFlow and Keras to implement 3D U-Net and Inception U-Net architectures.
    - Compile models with appropriate loss functions (e.g., Dice loss), optimizers (e.g., Adam), and evaluation metrics (Dice coefficient, accuracy).

3. **Training Procedure**:
    - Train models for 50 epochs with a defined batch size and a learning rate of 1e-3.

4. **Model Evaluation**:
    - Evaluate trained models using the validation dataset.
    - Compute metrics such as Dice coefficient and accuracy to gauge segmentation performance.

### Results and Discussion

The study evaluated the performance of U-Net and Inception U-Net architectures for segmenting brain tumors in MRI images.

- **3D U-Net**: Achieved a Dice coefficient of 0.5536 and an accuracy of 98.42%.
- **Inception U-Net**: Achieved a Dice coefficient of 0.6516 and an accuracy of 98.29%.

Despite the additional complexity in Inception U-Net with multi-scale feature extraction, its segmentation effectiveness was slightly lower than the standard U-Net.

### Sample of Predicted and Actual Masks

- **Inception U-Net**
- **U-Net**

## Conclusion

The U-Net architecture exhibited better segmentation performance compared to the Inception U-Net. The study's findings underscore the importance of evaluating different architectures and highlight the potential of deep learning models in improving brain tumor segmentation tasks.

---

### Files Included

- **images/**: Folder containing 3064 grayscale MRI images.
- **masks/**: Folder containing 3064 binary mask images indicating tumor regions.
- **scripts/**: Python scripts for data preprocessing, model implementation, training, and evaluation.
- **results/**: Folder containing model evaluation results, predicted masks, and comparison charts.
- **README.md**: This file providing an overview of the project.

### How to Run

1. **Setup**:
    - Install required libraries using `requirements.txt`.
    - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation).

2. **Data Preprocessing**:
    - Run `data_preprocessing.py` to preprocess the images and masks.

3. **Model Training**:
    - Run `train_unet.py` to train the U-Net model.
    - Run `train_inception_unet.py` to train the Inception U-Net model.

4. **Model Evaluation**:
    - Run `evaluate_model.py` to evaluate the trained models and generate results.

5. **Results Visualization**:
    - Check the `results/` folder for evaluation metrics and sample predicted masks.

### Contact

For any questions or issues, please contact [Your Name] at [Your Email].
