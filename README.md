# me-types-mapper

### Description:

_me-types-mapper_ is a python package that propose a probabilistic mapping between cell types from two different 
datasets based on shared morpho-electrical features. You can see an application here: 
https://www.biorxiv.org/content/10.1101/2021.11.24.469815v1


### Installation:

run `pip install git+https://github.com/BlueBrain/me-types-mapper.git`

### Examples:

`from me_types_mapper.mapper.coclustering_functions import cross_predictions_v2`
`alpha_opt, map_, c1, c2, dict_cluster_label, cluster_ref, fig_alpha, fig_d_opt = cross_predictions_v2(data_1, data_2, 
msk_ephys_, msk_morpho_, lbls)`

###  Funding & Acknowledgment:

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021 Blue Brain Project/EPFL
