# README: Medical Imaging Segmentation and Applications (MISA)
### Xavier Beltran Urbano and Frederik Hartmann
#### University of Girona

## Dataset
The dataset used in this study is the IBSR18, containing 18 T1-weighted scans of normal subjects from the Internet Brain Segmentation Repository (IBSR). It includes preprocessed scans with a 1.5 mm slice thickness and ground truth segmentation for white matter (WM), grey matter (GM), and cerebrospinal fluid (CSF).

## Methodology
Our approach employs an ensemble of 2D and 3D convolutional neural networks for brain tissue segmentation in MR images. We tested various architectures like U-Net, Res-U-Net, Multi-Resolution-U-Net, Dense-U-Net, and SegResNet. The dataset was divided into training, validation, and test sets. Preprocessing steps included normalization and data augmentation. We experimented with different input orientations (axial and coronal) and employed techniques like weighted categorical cross-entropy loss, learning rate scheduling, early stopping, and model selection based on validation loss.

## Results
### Single Model Results

| Model                     | CSF    | GM     | WM     | Mean  |
|---------------------------|--------|--------|--------|-------|
| 2D Coronal U-Net          | 0.878  | 0.937  | 0.933  | 0.917 |
| 2D Coronal Dense U-Net    | 0.899  | 0.937  | 0.938  | 0.925 |
| 2D Coronal Multi-U-Net    | 0.890  | 0.935  | 0.936  | 0.920 |
| 2D Coronal Res-U-Net      | 0.882  | 0.931  | 0.931  | 0.915 |
| 2D Axial U-Net            | 0.868  | 0.929  | 0.922  | 0.906 |
| 2D Axial Dense-U-Net      | 0.868  | 0.920  | 0.920  | 0.902 |
| 2D Axial Multi-U-Net      | 0.876  | 0.923  | 0.926  | 0.908 |
| 2D Axial Res-U-Net        | 0.866  | 0.925  | 0.921  | 0.904 |
| 2D Seg-Res-Net            | 0.877  | 0.933  | 0.935  | 0.915 |
| 3D U-Net                  | 0.882  | 0.942  | 0.942  | 0.922 |
| 3D Seg-Res-Net            | 0.888  | 0.935  | 0.937  | 0.921 |
| SynthSeg                  | 0.812  | 0.829  | 0.888  | 0.843 |

*Table: Single model results on the validation set.*

### Ensemble Results

| Model                               | CSF    | GM     | WM     | Mean  |
|-------------------------------------|--------|--------|--------|-------|
| The Coronal Ensemble Mean           | 0.895  | 0.939  | 0.939  | 0.925 |
| The Coronal Ensemble Maximum        | 0.893  | 0.939  | 0.939  | 0.923 |
| The Coronal Ensemble Majority       | 0.890  | 0.939  | 0.937  | 0.922 |
| The Axial Ensemble Mean             | 0.884  | 0.930  | 0.928  | 0.914 |
| The Axial Ensemble Maximum          | 0.881  | 0.930  | 0.927  | 0.913 |
| The Axial Ensemble Majority         | 0.877  | 0.930  | 0.925  | 0.911 |
| The Coronal + Axial Mean            | 0.897  | 0.939  | 0.938  | 0.925 |
| The Coronal + Axial Maximum         | 0.893  | 0.938  | 0.937  | 0.923 |
| The Coronal + Axial Majority        | 0.894  | 0.940  | 0.938  | 0.924 |
| The Multidimensional Ensemble Mean  | 0.904  | 0.945  | 0.948  | 0.932 |

*Table: Ensemble results on the validation set.*

## Conclusion
The study clearly illustrates the efficacy of an ensemble methodology that synergizes 2D and 3D convolutional neural networks (CNNs) for segmenting brain tissue. This innovative approach benefits significantly from leveraging various orientations of 2D slices, in combination with both 2D and 3D models. This multifaceted strategy enhances the accuracy of segmentation substantially. Among the various techniques explored, the ensemble method, especially the mean of probabilities technique, stands out for its exceptional robustness and precision in results.