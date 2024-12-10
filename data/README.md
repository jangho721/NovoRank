## How to use NovoRank

To use NovoRank for your datasets, you HAVE TO train your own model fitting to your datasets.

### Step 1. Preparation datasets
As an initial step, a user MUST make their datasets to fit NovoRank input standard.

#### De novo search result
Once you perform de novo search using any tools such as PEAKS, pNovo3 and DeepNovo, you MUST convert the result to below form:

Source File|Scan number|Peptide|Score
---|---|---|---|
Hela_1.mgf|10|HKPSVK|85|

Note that each column is separated by comma (comma-separated value format (CSV)).

#### Database search result
NovoRank generates positive and negative labels based on database search result from the same MS/MS spectra used in the de novo search. Therefore, it only needs for training. If a user uses pre-trained model, this file is not needed for the further step. After conducting database search, only reliable PSMs are prepared as below format:

Source File|Scan number|GT
---|---|---|
Hela_1.mgf|3|KPVGAAK| 

Note that each column is separated by comma (comma-separated value format (CSV)).

#### Note for post-translational modification notation
NovoRank assumes that all Cysteines (C) have a fixed modification Carbamidomethylation.
As a variable modification, it only allows an oxidation on Methionine as lower letter "m".
For example, if AM+15.99EENGR, a user must convert the sequence to AmEENGR.

### Step 3. Generation of deep learning input
Based on the results of both de novo search and MS-clust, NovoRank generates top two candidates.
The top two candidates are an initial point to train deep learning model.
