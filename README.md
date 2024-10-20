# EDPNet
This is the official repository to the paper "[EDPNet: An Efficient Dual Prototype Network for Motor Imagery EEG](https://arxiv.org/pdf/2407.03177v1)".

## Abstract
![image](https://github.com/hancan16/EDPNet/blob/main/figs/frameworkl.png)
- Inspired by clinical prior knowledge of EEG-MI and human brain recognition mechanisms, we propose a high performance, lightweight, and interpretable MI-EEG decoding model EDPNet. The EDPNet simultaneously considers and overcomes three major challenges in MI-BCIs.
- To extract highly discriminative features from EEG signals, we design two novel modules, ASSF and MVP, for the feature extractor of EDPNet. The ASSF module extracts effective spatial-spectral features, and the MVP module extracts powerful multi-scale temporal features.
- To overcome the small-sample issue of MI tasks, we propose a novel DPL approach to optimize the distribution of features and prototypes, aiming to obtain a robust feature space. This enhances the generalization capability and classification performance of our EDPNet.
- We conduct experiments on three benchmark public datasets to evaluate the superiority of the proposed EDPNet against state-of-the-art (SOTA) MI decoding methods Additionally, comprehensive ablation experiments and visual analysis demonstrate the effectiveness and interpretability of each module in the proposed EDPNet.

## Requirements:
- Python 3.10
- Pytorch 2.12

## Rusults and Visualization
In the following datasets we have used the official criteria for dividing the training and test sets:
- [BCI_competition_IV 2a](https://www.bbci.de/competition/iv/) -acc 84.11%
- [BCI_competition_IV 2b](https://www.bbci.de/competition/iv/) -acc 86.65%
- [BCI_competition_III IVa](https://bbci.de/competition/iii/desc_IVa.html) -acc 82.03%

![image](https://github.com/hancan16/EDPNet/blob/main/figs/tsne.png)

## Acknowledgments
We are deeply grateful to Martin for providing clear and easily executable code in the [channel-attention](https://github.com/martinwimpff/channel-attention) repository. In our paper, we referenced the code and results from [channel-attention](https://github.com/martinwimpff/channel-attention) to ensure the reliability of our reproductions of the baseline methods.


## Contact
If you have any questions, please feel free to email hancan@sjtu.edu.cn.