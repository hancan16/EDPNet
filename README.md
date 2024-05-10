# EDPNet
This is the official repository to the paper "EDPNet: An Efficient Dual Prototype Network for Motor Imagery Decoding"

## Abstract
- We propose an end-to-end deep learning architecture for MI decoding, which is able to effectively extract discriminative spatial-spectral features and capture more comprehensive long-term temporal features.
- To mitigate the intra-class variation of EEG signals and the small sample problem of MI tasks, we proposed a novel dual prototype learning approach to further enhance the generalization capability and recognition performance of the model.
- We conduct experiments on three benchmark public datasets to evaluate the superiority of the proposed method against state-of-the-art (SOTA) MI decoding methods. Additionally, comprehensive ablation experiments have validated the effectiveness and interpretability of each module in our proposed method.

## Requirements:
- Python 3.10
- Pytorch 2.12

## Datasets
In the following datasets we have used the official criteria for dividing the training and test sets:
- [BCI_competition_IV 2a](https://www.bbci.de/competition/iv/) -acc 83.87%
- [BCI_competition_IV 2b](https://www.bbci.de/competition/iv/) -acc 86.60%
- [BCI_competition_III](https://bbci.de/competition/iii/desc_IVa.html) IVa -acc 80.98%