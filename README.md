# NovoRank: Refinement for De Novo Peptide Sequencing Based on Spectral Clustering and Deep Learning

NovoRank is a post-processing tool designed to improve the accuracy of de novo peptide sequencing in proteomics. Unlike database-dependent methods, de novo sequencing derives peptide sequences directly from tandem mass spectrometry (MS/MS) data, enabling the discovery of novel peptides. However, reliance on incomplete scoring functions often leads to incorrect identifications. NovoRank addresses this by re-ranking candidate peptides to recover correct identifications, enhancing both precision and recall. By reassigning the optimal peptide from diverse de novo sequencing results, NovoRank offers a robust solution for overcoming the noise and ambiguities inherent in MS/MS data.

For detailed insights behind NovoRank, refer to the NovoRank paper.

<br>

<p align="center">
    <img src="assets/img.png" width="700"\>
</p>

<hr>

## Requirements

NovoRank is implemented and tested with the following dependencies:

### Software and Libraries:
- Python == 3.9
- [DeepLC](https://github.com/compomics/DeepLC)
- [MS-Cluster](http://proteomics.ucsd.edu/software-tools/ms-clusterarchives)
- CometX ( In-house software modified to calculate XCorr, based on Comet software )
<br>

### To install the required Python packages:

1. Clone the repository or download the code.
2. Install the dependencies listed in `requirements.txt`:

```c
pip install -r requirements.txt
```

## Datasets

All datasets used in this work are available for download from [Zenodo](https://zenodo.org/records/14046459).

## Parameters

## Citation
```bibTeX
@article{cho2024transformer,
  title = {NovoRank: Refinement for De Novo Peptide Sequencing Based on Spectral Clustering and Deep Learning},
  shorttitle = {NovoRank},
  author = {Seo, Jangho and Choi, Seunghyuk and Paek, Eunok},
  journal={},
  year={}
}
```

## License
<pre>
- NovoRank © 2024 is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
  This license requires that reusers give credit to the creator. It allows reusers to distribute, 
  remix, adapt, and build upon the material in any medium or format, for noncommercial purposes only. 
  If others modify or adapt the material, they must license the modified material under identical terms.
</pre>

## Contact
If you have any questions, feel free to [open an issue](https://github.com/poloclub/transformer-explainer/issues/new/choose) or contact [Jangho Seo](https://jangho721.github.io/) or any of the contributors listed above.
