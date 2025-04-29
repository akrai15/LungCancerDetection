🫁 Lung Cancer Detection using MHACNN (Multi-Head Attention CNN)

🔬 Overview

MHACNN (Multi-Head Attention Convolutional Neural Network) is a novel deep learning framework for automated lung cancer detection using both CT scan and histopathological images. It synergizes CNN’s spatial feature extraction power with a multi-head attention mechanism to enhance diagnostic accuracy and interpretability.

    Achieved 98% accuracy on the LC25000 dataset and 97% accuracy on the LUNA16 dataset.

📊 Key Features

    Dual-dataset compatibility: LUNA16 (CT scans) and LC25000 (Histopathology)

    ![image](https://github.com/user-attachments/assets/72318fbc-b651-4503-9a72-48e4684568a5)


    Hybrid architecture: CNN + Multi-Head Attention

    Robust preprocessing and augmentation pipelines

    Interpretability via Grad-CAM visualizations

    Strong generalization, minimized overfitting

🧠 Model Architecture (MHACNN)

MHACNN Architecture <!-- Make sure this image exists or upload one -->

    Convolutional layers with progressive filters (32 → 64 → 128)

    Batch Normalization and Dropout

    Multi-Head Attention layers for capturing long-range dependencies

    Dense layers for binary classification (Benign vs Malignant)
    ![image](https://github.com/user-attachments/assets/130eba65-3986-4632-8762-ae14198b26fb)


📁 Datasets Used
1. LUNA16

    888 low-dose CT scans from the LIDC-IDRI database

    1,186 annotated lung nodules

    Used for segmentation and detection tasks

2. LC25000

    25,000 histopathological images (Benign, Adenocarcinoma, Squamous cell carcinoma)

    High-resolution RGB images for classification





Features

    Multi-Modal Detection: Works with both CT and histopathological images.

    Multi-Head Attention: Integrates attention mechanisms to focus on critical image regions, improving interpretability and diagnostic precision.

    High Accuracy: Achieves 97% accuracy on LUNA16 and 98% on LC25000.

    Generalizable: Robust across different imaging modalities and datasets.

    Open Source: Code and research paper available for reproducibility and further research.

Methodology

MHACNN combines the hierarchical feature extraction capabilities of CNNs with the ability of multi-head attention to focus on the most relevant regions of an image. The workflow is as follows:

    Preprocessing: Input images (CT or histopathology) are normalized and resized.

    Feature Extraction: CNN layers extract spatial features.

    Multi-Head Attention: Attention layers focus on salient regions, mimicking expert radiologists’ focus.

    Classification: Dense layers output the probability of cancerous vs. non-cancerous tissue.



The MHACNN model consists of:

    CNN Backbone: Extracts hierarchical features from input images.

    Multi-Head Attention Module: Multiple attention heads focus on different image regions, capturing long-range dependencies and subtle patterns.

    Dense Layers: Aggregate features and perform final classification.

Key Advantages:

    Improved accuracy and generalizability across modalities.

    Reduced overfitting via dynamic attention.

    Enhanced feature capture and interpretability.

    ![image](https://github.com/user-attachments/assets/d9999ce4-6042-4fee-bf6d-64b66b4ed044)




🚀 Training
For LC25000 Dataset:

python training/train_lc25000.py

For LUNA16 Dataset:

python training/train_luna16.py

Training settings:

    LC25000: 100 epochs

    LUNA16: 500 epochs

📈 Results
Dataset	Accuracy	Precision	Recall	F1-Score
LC25000	98%	0.97	1.00	0.99
LUNA16	97%	0.88	0.84	0.86

🔍 Visualizations (Grad-CAM)

Grad-CAM is used to visualize the regions most influential in the model’s decisions.
      ![image](https://github.com/user-attachments/assets/ae7c6cdb-048e-4c2f-8589-9abd8fe45563)
![image](https://github.com/user-attachments/assets/d2baeaec-7fbd-45d4-afb1-e5353552fe07)


The model outperforms traditional CNNs and transfer learning approaches, demonstrating consistent performance across both CT and histopathological data. The attention mechanism improves both diagnostic accuracy and model interpretability, aiding radiologists in early-stage lung cancer diagnosis
