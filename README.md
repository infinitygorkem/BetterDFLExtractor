# BetterDFLExtractor

## Introduction
BetterDFLExtractor is designed as a more efficient and powerful alternative to the standard extractor in DeepFaceLab. It aims to improve the face extraction process used in deepfake creation, focusing on enhanced performance and accuracy.

## Acknowledgments
This project incorporates developments from the `3DDFA_V2` project. Special thanks to Jianzhu Guo, Xiangyu Zhu, Yang Yang, Fan Yang, Zhen Lei, and Stan Z Li for their significant contributions to the field of 3D face alignment.

Relevant references:
- Guo, J., Zhu, X., Yang, Y., Yang, F., Lei, Z., & Li, S. Z. (2020). Towards Fast, Accurate and Stable 3D Dense Face Alignment. Proceedings of the European Conference on Computer Vision (ECCV).
- Guo, J., Zhu, X., & Lei, Z. (2018). 3DDFA. Retrieved from [3DDFA GitHub Repository](https://github.com/cleardusk/3DDFA)

## License
This project is licensed under the [MIT License](LICENSE.md).

## Installation
1. Clone this repository into your DeepFaceLab directory:
   - For Windows: `_internal/DeepFaceLab`
   - For Linux: `DeepFaceLab_Linux/DeepFaceLab`
2. Use Anaconda to create an environment using the `environment.yaml` file provided.
3. Run `caller.py` in the DeepFaceLab folder to launch the extractor.

## Notes
Please note that BetterDFLExtractor is an amateur product and is subject to further optimization. Future plans include:
- Development of a custom-made alignment model.
- Implementation of segmentation using U-Net models.
- Extensive performance optimizations.

Feedback and contributions to enhance and optimize this tool are greatly welcomed.
