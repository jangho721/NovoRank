# NovoRank: Machine Learning-Based Post-Processing Tool for De Novo Peptide Sequencing

NovoRank is a post-processing tool designed to improve the accuracy of de novo peptide sequencing in proteomics. Unlike database-dependent methods, de novo sequencing derives peptide sequences directly from tandem mass spectrometry (MS/MS) data, enabling the discovery of novel peptides. However, reliance on incomplete scoring functions often leads to incorrect identifications. NovoRank addresses this by re-ranking candidate peptides to recover correct identifications, enhancing both precision and recall. By reassigning the optimal peptide from diverse de novo sequencing results, NovoRank offers a robust solution for overcoming the noise and ambiguities inherent in MS/MS data.

For detailed insights behind NovoRank, refer to the [NovoRank paper]().

<br>

<p align="center">
    <img src="assets/img.png" width="700"\>
</p>
<hr>

## Overview
#### Code structure
``` Unicode
NovoRank
  │ 
  ├── generate_candidates_and_extract_features.py
  │ 
  ├── src
  │    │     
  │    ├── finetuned_models: BERT finetuned on downstream tasks
  │    │      
  │    ├── lib
  │    │     ├── agent.py: code for agent
  │    │     ├── memory.py: code for memory
  │    │     └── reward.py: code for reward
  │    │  
  │    ├── script: script for training and evaluating a pruned BERT on downstream tasks
  │    │ 
  │    ├─── utils
  │    │     ├── default_param.py: default cfgs
  │    │     └── utils.py: utility functions
  │    │ 
  │    ├─── transformers: refer to https://github.com/huggingface/transformers/
  │    │     
  │    ├─── main.py: main file to run Automated-BERT-Regularization
  │    │    
  │    └─── train.py: code for training the agent
  │
  ├─── models: GLUE data
  │
  ├─── pretrained: GLUE data
  │
  └─── software: shell scripts for demo
```

## Requirements

NovoRank is implemented and tested with the following dependencies:
### Software and Libraries:
- Python == 3.9
- [DeepLC](https://github.com/compomics/DeepLC)
- [MS-Cluster](http://proteomics.ucsd.edu/software-tools/ms-clusterarchives)
- [CometX](https://github.com/jangho721/NovoRank/tree/main/software/CometX) (In-house software modified to calculate XCorr, based on Comet software)
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
<pre>
- Training.Sample=
  Specify gated reference sample files for gating strategy learning (comma separated value, CSV format)
  Make sure that the CSV files have a column named 'Label' in the header, where cell labels are written.
  Multiple files can be specified by mulitple lines below.
  ex)
   Training.Sample= E:\cytof\reference_gating1.csv
   Training.Sample= E:\cytof\reference_gating2.csv

- Training.UngatedCellLabel=
  Specify label for UNGATED cells
  ex)
   Training.UngatedCellLabel= NA
	   
- Data.Sample=
  Specify sample files or directory for automatic gating (CSV format)
  Given a directory, all files in it are gated.
  Multiple files can be specified by mulitple lines below.
  ex)
   Data.Sample= E:\cytof\data1.csv
   Data.Sample= E:\cytof\data2.csv
   Data.Sample= E:\cytof\data3     #possible to specify directory
</pre>

## Usage
<pre>
- Command: java -jar CyGate.jar --c configFile
- Example: java -jar CyGate.jar --c foo.txt
</pre>

## Results
<pre>
- For the files specified in Data.Sample, *_cygated.csv files are generated.
- The gating results are added to the last column, named 'Gated'.
</pre>

## Credits
NovoRank is created by <a href="https://jangho721.github.io/" target="_blank">Jangho Seo</a>, Seunghyuk Choi, and Eunok Paek at the Hanyang University.

## Citation
```bibTeX
@article{sep2024novorank,
  title = {NovoRank: Refinement for De Novo Peptide Sequencing Based on Spectral Clustering and Deep Learning},
  shorttitle = {NovoRank},
  author = {Seo, Jangho and Choi, Seunghyuk and Paek, Eunok},
  journal={Journal of Proteome Research},
  year={2024}
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
If you have any questions, feel free to [open an issue](https://github.com/jangho721/NovoRank/issues/new) or contact [Jangho Seo](https://jangho721.github.io/) or any of the contributors listed above.
