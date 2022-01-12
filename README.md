# me_types_mapper

Description:

_me-typs-mapper_ is a python package that propose a probabilistic mapping between cell types from two different datasets 
based on shared morpho-electrical features.

Installation:

run `pip install git+https://github.com/BlueBrain/me-types-mapper.git`

Examples:

`from me_types_mapper.mapper.coclustering_functions import cross_predictions_v2`
`alpha_opt, map_, c1, c2, dict_cluster_label, cluster_ref, fig_alpha, fig_d_opt = cross_predictions_v2(data_1, data_2, 
msk_ephys_, msk_morpho_, lbls)`
