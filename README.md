# euphemism_shared_task_emnlp2022
The Transformers for euphemism detection baseline (TEDB) System to a Shared Task on Euphemism Detection @EMNLP-2022
https://codalab.lisn.upsaclay.fr/competitions/5726.

Our best submission based on RoBERTa tweet 124m + KimCNN (0.816 test F1-score) can be viewed as a colab file at https://colab.research.google.com/drive/127DHVVNvcl1RAdSBZeVwGmpZYc0wSlC8?usp=sharing.

Presentation slides: https://docs.google.com/presentation/d/11ufCfDBz9AK6h_YU7s6KbP_P0Ceeq8jcYSoauwfinlo/edit?usp=sharing

## Training and Testing
Copy the [Euphemism22_code](Euphemism22_code) folder to any colab directory and mount the codebase folder to the node. The jupyter notebooks, such as [Euphemism22_cardiffnlp_twitter_roberta_base_sentiment_latest_kim_cnn.ipynb](Euphemism22_cardiffnlp_twitter_roberta_base_sentiment_latest_kim_cnn.ipynb), provide the complete steps and pipelines. 

## Cite this paper
A link to the [paper](https://aclanthology.org/2022.flp-1.pdf#page=15) and its [ArXiv](https://arxiv.org/abs/2301.06602).

### Citation
Wiriyathammabhum, P. (2022, December). TEDB System Description to a Shared Task on Euphemism Detection 2022. In Proceedings of the 3rd Workshop on Figurative Language Processing (FLP) (pp. 1-7).

``` bibtex
@inproceedings{wiriyathammabhum2022tedb,
  title={TEDB System Description to a Shared Task on Euphemism Detection 2022},
  author={Wiriyathammabhum, Peratham},
  booktitle={Proceedings of the 3rd Workshop on Figurative Language Processing (FLP)},
  pages={1--7},
  year={2022}
}
```
