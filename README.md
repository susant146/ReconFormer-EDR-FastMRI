# ReconFormer-EDR-FastMRI
This repository provides the source code MRI reconstruction. ReconFormer-EDR: An encoder-decoder-refined network within a recurrent transformer architecture for high-quality MRI image reconstruction even at a higher acceleration rate.

## Abstract
Accelerated magnetic resonance imaging (MRI) reconstruction is a challenging and ill-posed inverse problem due to severe $k$-space undersampling. In this paper, we propose ReconFormer-EDR, a light-weight encoder-decoder-refined network within a recurrent transformer architecture for high-quality image reconstruction even at a higher acceleration rate. Based on the recurrent pyramid transformer layer (RPTL), ReconFormer-EDR integrates channel-wise attention and residual blocks in both the encoder and decoder stages, followed by an optimized data consistency layer. The model employs a consistent gradient loss to enhance reconstruction quality and preserve important image features. Evaluated on the fastMRI dataset across different sequences, ReconFormer-EDR outperforms several state-of-the-art methods, including the baseline ReconFormer model. It also exhibits strong generalization to limited and unseen training data, showing promise for robust and efficient MRI reconstruction under challenging conditions.

## Block Diagram
<img width="3247" height="779" alt="BD_ReconFormerEDR" src="https://github.com/user-attachments/assets/2ec44711-5823-46e2-ae7f-b2731b6f1144" />


