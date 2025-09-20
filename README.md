# ReconFormer-EDR-FastMRI
This repository provides the source code MRI reconstruction. ReconFormer-EDR: An encoder-decoder-refined network within a recurrent transformer architecture for high-quality MRI image reconstruction even at a higher acceleration rate.

## Abstract
Accelerated magnetic resonance imaging (MRI) reconstruction is a challenging and ill-posed inverse problem due to severe $k$-space undersampling. In this paper, we propose ReconFormer-EDR, a light-weight encoder-decoder-refined network within a recurrent transformer architecture for high-quality image reconstruction even at a higher acceleration rate. Based on the recurrent pyramid transformer layer (RPTL), ReconFormer-EDR integrates channel-wise attention and residual blocks in both the encoder and decoder stages, followed by an optimized data consistency layer. The model employs a consistent gradient loss to enhance reconstruction quality and preserve important image features. Evaluated on the fastMRI dataset across different sequences, ReconFormer-EDR outperforms several state-of-the-art methods, including the baseline ReconFormer model. It also exhibits strong generalization to limited and unseen training data, showing promise for robust and efficient MRI reconstruction under challenging conditions.

## Block Diagram
<img width="3247" height="779" alt="BD_ReconFormerEDR" src="https://github.com/user-attachments/assets/2ec44711-5823-46e2-ae7f-b2731b6f1144" />
**Fig.1:** Block diagram of the proposed ReconFormer-EDR architecture. (a) Recurrent units (RU) are updated through deep unrolling across multiple iterations. (b) The attention guided encoder-decoder and refine module (RM) used in the RU blocks. (c) The illustration of a recurrent unit (RU). Here, the ReconFormer Block (RFB) is same as defined in [ref-1].

## Results
**Qualitative Results** <br>
![Acc4_QualityComp1](https://github.com/user-attachments/assets/1a90ef2b-776a-411a-bb9a-b129e9674c1b) <br>
**Fig.2** Acceleration factor R = 4: Qualitative assessment of different methods on the multicoil fatMRI dataset. The second row of each subplot shows the corresponding ×4 magnified patch corresponds to green rectangle region. The yellow arrow indicates the fine details that ReconFormer-EDR preserves better compared to other methods.<br>

![Acc8_QualityComp1](https://github.com/user-attachments/assets/69edf60e-66f4-4719-bae2-7b19799bd6d5) <br>
**Fig.3** Acceleration factor R = 8: Qualitative assessment of different methods on the multicoil fatMRI dataset. The second row of each subplot shows the corresponding ×4 magnified patch corresponds to green rectangle region. The yellow arrow indicates the fine details that ReconFormer-EDR preserves better compared to other methods. <br>

**Quantitative Results** <br>
**Table.1**: Quantitative comparison of various state-of-the-art methods with the proposed ReconFormer-EDR model. The best and second performance measure are highlighted in Bold and Underline, respectively. <br>
<img width="1030" height="270" alt="image" src="https://github.com/user-attachments/assets/207b9bf7-9475-4601-a0d2-68647b293925" />

# Supplementary Material 
we have includedd the 95% confidence intervals (CIs) computed volume-wise for PSNR, SSIM, and NMSE. For the fastMRI knee dataset, the 90% CI for PSNR is approximately ±0.15 dB and for SSIM ±0.004, for the proposed ReconFormer-EDR model. Furthermore, we now provide a per-sequence breakdown (CORPD vs. CORPDFS) to highlight performance across different contrasts in Table.2 and Table.3. To address concern regarding data partitioning, we have clarified in Section III-A that training and validation sets follow the fastMRI patient-wise split [2]. Note that training was performed on approximately 15K slices of the training set and 20% of these was used for validation. The results in Table I are reported on the remaining 3K slices of the validation set. <br>

**Table.2**: R=4: QUANTITATIVE COMPARISON OF VARIOUS ARCHITECTURES WITH THE PROPOSED RECONFORMER-EDR MODEL. THE BEST PERFORMANCE MEASURES ARE HIGHLIGHTED IN BOLD FONT.
<img width="1087" height="396" alt="image" src="https://github.com/user-attachments/assets/e38ac850-7484-4029-9346-6ce3ea89f3c4" /> <br>

**Table.3**: R=8: QUANTITATIVE COMPARISON OF VARIOUS ARCHITECTURES WITH THE PROPOSED RECONFORMER-EDR MODEL. THE BEST PERFORMANCE MEASURES ARE HIGHLIGHTED IN BOLD FONT.
<img width="1091" height="389" alt="image" src="https://github.com/user-attachments/assets/591e5acc-71c7-4716-93f1-3b903a290fe3" />


# Dataset
The code was trained, validated and tested on [fastMRI](https://fastmri.med.nyu.edu/) Knee multicoil dataset. 

# Cite
If you are using our code, please cite the following paper: <br>
Panigrahi, S. K., Sasmal, P., Dewan, D., & Sheet, D. (2025). Encoder-Decoder Refined Recurrent Transformer Network for Accelerated MRI Reconstruction. 4th IEEE CVMI-2025. [Accepted]

## Reference
1. Guo, Pengfei, et al. "Reconformer: Accelerated mri reconstruction using recurrent transformer." IEEE transactions on medical imaging 43.1 (2023): 582-593.
2. Knoll, Florian, et al. "fastMRI: A publicly available raw k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning." Radiology: Artificial Intelligence 2.1 (2020): e190007.
