# ERA-V3 Session 8: Advanced Neural Network Architectures

This repository implements a custom CNN architecture for CIFAR10 classification, adhering to specific architectural requirements.

## Network Architecture

The network follows a C1C2C3C4 architecture, where each block contains exactly three convolutions, with the last one using a stride of 2 for downsampling.

### Architecture Breakdown:

- **C1 Block: Regular Convolutions**
  - Conv1.1: 3×3, channels: 3→24
  - Conv1.2: 3×3, channels: 24→32
  - Conv1.3: 3×3, stride=2, channels: 32→48

- **C2 Block: Depthwise Separable Convolutions**
  - DWConv2.1: 3×3, channels: 48→48, dropout=0.05
  - DWConv2.2: 3×3, channels: 48→48, dropout=0.05
  - DWConv2.3: 3×3, stride=2, channels: 48→64, dropout=0.05

- **C3 Block: Dilated Convolutions with Skip Connections**
  - Conv3.1: 3×3, dilation=2, groups=4, channels: 64→64, dropout=0.15
  - Conv3.2: 3×3, dilation=4, groups=4, channels: 64→64, dropout=0.15
  - Conv3.3: 3×3, stride=2, channels: 64→72, dropout=0.15

- **C4 Block: Grouped Convolutions with Skip Connections**
  - Conv4.1: 3×3, groups=4, channels: 72→72, dropout=0.15
  - Conv4.2: 3×3, groups=4, channels: 72→72, dropout=0.15
  - Conv4.3: 3×3, stride=2, channels: 72→88, dropout=0.15

### Output:
- Global Average Pooling
- Dropout(0.2)
- Fully Connected Layer: 88→10 classes

## Model Training Instructions

### Requirements
To install the necessary packages, run:
```
pip install -r requirements.txt
```

### Training the Model
To train the model, execute the following command:
```bash
python src/train.py
```
Make sure to adjust any parameters in the `train.py` file as needed for your specific training setup.

## Meeting Architecture Requirements

1. **C1C2C3C4 with 3 Convolutions**:
   - Each block contains exactly 3 convolutions.
   - The last convolution in each block uses a stride of 2.
   - Progressive channel growth: 3→48→64→72→88.

2. **Receptive Field > 44**:
   - Final RF: 153×153.
   - Achieved through:
     - Regular convolutions
     - Dilated convolutions (d=2,4 in C3)
     - Strided convolutions
     - Accumulated jump factors

3. **Depthwise Separable Convolution**:
   - Implemented in the C2 block.
   - Reduces parameters while maintaining performance.
   - Light dropout (0.05) for regularization.

4. **Dilated Convolution**:
   - Implemented in the C3 block.
   - First conv: dilation=2.
   - Second conv: dilation=4.
   - Skip connections for better gradient flow.

5. **GAP and FC**:
   - Global Average Pooling after the C4 block.
   - Dropout(0.2) for regularization.
   - Single Fully Connected layer (88→10).

## Training Features

1. **Learning Rate Strategy**:
   - Layer-wise learning rates:
     - C1: LR * 2.0 (faster early learning)
     - C2: LR * 1.5
     - C3: LR * 1.0
     - C4: LR * 0.8
     - FC: LR * 1.5

2. **Regularization**:
   - Progressive dropout:
     - C1: No dropout
     - C2: 0.05
     - C3/C4: 0.15
     - Final: 0.2
   - Skip connections in C3/C4.
   - Weight decay: 1e-5.

3. **Architecture Features**:
   - Grouped convolutions (groups=4) in C3/C4.
   - Dilated convolutions in C3.
   - Depthwise separable in C2.
   - Skip connections for better gradient flow.

4. **Optimization**:
   - Adam optimizer.
   - OneCycleLR scheduler.
   - Base LR: 0.002.
   - Max LR: 0.02.
   - PCT_START: 0.1.

## Training Summary

|   Epoch |   Train Loss |   Train Acc(%) |   Test Loss |   Test Acc(%) |
|---------|--------------|----------------|-------------|----------------|
|       1 |       1.6844 |          37.36 |      1.3356 |         51.07  |
|       2 |       1.3791 |          50.01 |      1.1749 |         56.45  |
|       3 |       1.2198 |          56.07 |      0.9687 |         65.42  |
|       4 |       1.1236 |          59.86 |      0.8662 |         68.79  |
|       5 |       1.0481 |          62.93 |      0.7939 |         71.69  |
|       6 |       0.9942 |          65.01 |      0.7425 |         73.89  |
|       7 |       0.9476 |          66.92 |      0.724  |         74.56  |
|       8 |       0.9216 |          67.73 |      0.7065 |         75.25  |
|       9 |       0.8928 |          68.84 |      0.7308 |         75.17  |
|      10 |       0.8681 |          69.58 |      0.6365 |         78.2   |
|      11 |       0.8465 |          70.38 |      0.5924 |         79.82  |
|      12 |       0.8261 |          71.19 |      0.6208 |         78.12  |
|      13 |       0.8174 |          71.44 |      0.5577 |         80.14  |
|      14 |       0.8037 |          72.07 |      0.5867 |         79.56  |
|      15 |       0.7874 |          72.74 |      0.5583 |         80.92  |
|      16 |       0.7805 |          73.01 |      0.562  |         80.47  |
|      17 |       0.7615 |          73.64 |      0.5395 |         81.6   |
|      18 |       0.7545 |          73.82 |      0.5677 |         80.39  |
|      19 |       0.7479 |          74.22 |      0.5365 |         81.79  |
|      20 |       0.7381 |          74.43 |      0.5189 |         82.36  |
|      21 |       0.735  |          74.76 |      0.5189 |         82.3   |
|      22 |       0.726  |          74.97 |      0.5166 |         81.77  |
|      23 |       0.7199 |          75.26 |      0.5089 |         82.58  |
|      24 |       0.7116 |          75.39 |      0.5088 |         82.52  |
|      25 |       0.7051 |          75.68 |      0.505  |         82.7   |
|      26 |       0.6999 |          75.86 |      0.4902 |         82.69  |
|      27 |       0.6946 |          75.97 |      0.5287 |         82.17  |
|      28 |       0.6907 |          76.11 |      0.4977 |         82.58  |
|      29 |       0.6868 |          76.58 |      0.474  |         83.38  |
|      30 |       0.6816 |          76.52 |      0.5208 |         82.21  |
|      31 |       0.6759 |          76.76 |      0.4838 |         83.45  |
|      32 |       0.6736 |          76.82 |      0.4718 |         83.74  |
|      33 |       0.6739 |          76.66 |      0.4715 |         83.85  |
|      34 |       0.6653 |          76.94 |      0.4841 |         83.3   |
|      35 |       0.6622 |          77.15 |      0.469  |         83.88  |
|      36 |       0.6622 |          77.07 |      0.4527 |         84.36  |
|      37 |       0.6534 |          77.58 |      0.4712 |         83.85  |
|      38 |       0.6557 |          77.6  |      0.4822 |         83.49  |
|      39 |       0.6469 |          77.81 |      0.4645 |         84.13  |
|      40 |       0.6454 |          77.8  |      0.4491 |         84.54  |
|      41 |       0.648  |          77.67 |      0.4407 |         84.83  |
|      42 |       0.6417 |          77.86 |      0.4711 |         83.7   |
|      43 |       0.6409 |          77.86 |      0.4726 |         83.44  |
|      44 |       0.636  |          78.11 |      0.4426 |         84.93  |
|      45 |       0.6379 |          78.34 |      0.4461 |         84.72  |
|      46 |       0.6259 |          78.41 |      0.4648 |         84.21  |
|      47 |       0.6335 |          78.3  |      0.4584 |         84.51  |
|      48 |       0.6367 |          78.21 |      0.456  |         84.23  |
|      49 |       0.6245 |          78.45 |      0.4791 |         83.48  |
|      50 |       0.622  |          78.52 |      0.424  |         85.41  |
|      51 |       0.6237 |          78.34 |      0.4322 |         85.17  |
|      52 |       0.6196 |          78.48 |      0.4264 |         85.47  |
|      53 |       0.6242 |          78.61 |      0.4385 |         85.06  |
|      54 |       0.6205 |          78.6  |      0.4231 |         85.36  |
|      55 |       0.6114 |          78.96 |      0.4228 |         85.6   |
|      56 |       0.6125 |          79.01 |      0.4577 |         84.37  |
|      57 |       0.6175 |          78.77 |      0.4253 |         85.39  |
|      58 |       0.6096 |          78.98 |      0.436  |         85.1   |
|      59 |       0.61   |          79.05 |      0.4315 |         85.26  |
|      60 |       0.6086 |          79.02 |      0.4303 |         84.78  |
|      61 |       0.6006 |          79.32 |      0.4203 |         85.33  |
|      62 |       0.6035 |          79.14 |      0.4526 |         84.62  |
|      63 |       0.6027 |          79.2  |      0.4095 |         85.72  |
|      64 |       0.6039 |          79.25 |      0.4548 |         84.37  |
|      65 |       0.6058 |          79.24 |      0.4478 |         84.38  |
|      66 |       0.5967 |          79.51 |      0.4203 |         85.51  |
|      67 |       0.6001 |          79.38 |      0.4177 |         85.79  |
|      68 |       0.5966 |          79.55 |      0.4418 |         84.8   |
|      69 |       0.601  |          79.25 |      0.4021 |         86.01  |
|      70 |       0.601  |          79.38 |      0.4483 |         84.59  |
|      71 |       0.5949 |          79.63 |      0.4179 |         85.89  |
|      72 |       0.5924 |          79.5  |      0.4166 |         85.74  |
|      73 |       0.5857 |          79.92 |      0.4491 |         84.72  |
|      74 |       0.589  |          79.65 |      0.4232 |         85.59  |
|      75 |       0.5916 |          79.66 |      0.3927 |         86.58  |

**Best Test Accuracy: 86.58%**