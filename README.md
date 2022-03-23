# similarity_search
Reverse image search, similarity models, and image retrieval

This repository condains a PyTorch implementation of Grafit, which is based on this [BYOL](https://github.com/lucidrains/byol-pytorch) implementation. In addition to adding the KNN loss from Grafit, this implementation also uses [`Albumentation`](https://github.com/albumentations-team/albumentations) for augmentation. 

Grafit TODO: Create KNN tensor, project cosine similarity operation, make sure it's detached, build KNN loss. Define list of augs, build initial embedding loop, build embedding update for batch, enhance instance loss with N augmentations