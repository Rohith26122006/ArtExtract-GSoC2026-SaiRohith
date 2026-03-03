# ArtExtract GSoC 2026 - Task 1 Submission

##  Project Overview
This repository contains my submission for the **ArtExtract** project evaluation test under the **HumanAI** umbrella organization for **Google Summer of Code 2026**.

###  Task 1: CNN-RNN for Art Classification
Build a convolutional-recurrent architecture to classify paintings by:
- **Artist** (23 classes)
- **Genre** (10 classes)  
- **Style** (27 classes)

##  Architecture
- **CNN Backbone**: ResNet50 (pretrained on ImageNet)
- **Sequence Modeling**: Bidirectional LSTM (2 layers)
- **Attention Mechanism**: Spatial attention to focus on relevant regions
- **Multi-task Learning**: Simultaneous artist, genre, style classification

##  Dataset
- **WikiArt Dataset** (refined version)
- Training samples: 13,346 (artist file size)
- Validation samples: 5,706
- Handled CSV mismatches (artist file was smallest)

##  Training Results
- Trained on **2,000 samples** for **3 epochs** (due to hardware constraints)
- Outlier detection threshold: **0.3 confidence**
- **Outliers found**: 500 paintings with low confidence (<0.3) and wrong predictions
- **Pattern detected**: Artist Class 4 consistently misclassified as Class 22

##  Outlier Detection Methodology
1. Calculate prediction confidence (softmax probability)
2. Identify samples with confidence < 0.3 AND wrong prediction
3. Flag these as potential mislabeled paintings or atypical works

##  Repository Structure
