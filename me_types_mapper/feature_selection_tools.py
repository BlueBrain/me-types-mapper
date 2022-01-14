# This file is part of me-types-mapper.
#
#
# Copyright © 2021 Blue Brain Project/EPFL
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the APACHE-2 License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

import os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def unique_elements(array):
    """Return a list of unique elements of an array."""
    unique = []
    for x in array:
        if x not in unique:
            unique.append(x)
    return unique

def count_elements(array):
    """Return as a pandas DataFrame unique elements and the associated counts of an array."""
    
    unq = unique_elements(array)
    
    return pd.DataFrame([len(array[[y==x for y in array]]) for x in unq], index=unq, columns=['counts'])

def convert_edict_to_dataframe(edict):

    pd_dfs = []
    for cell in edict:
        values_vec = []
        column_names = []
        for protocol in edict[cell]:
            for dict_tmp in edict[cell][protocol]["soma"]:
                column_names.append(dict_tmp["feature"] + "|" + protocol)
                values_vec.append(dict_tmp["val"][0])
        pd_dfs.append(pd.DataFrame(values_vec, index=column_names, columns=[cell]))

    return pd.concat(pd_dfs, axis=1).T

dict_layer = {'layer 1': 'L1', 'L1': 'L1', 'layer 2/3': 'L23', 'L2/3': 'L23', 'L23': 'L23',
'layer 4': 'L4', 'L4': 'L4', 'layer 5': 'L5', 'L5': 'L5',
'layer 6': 'L6', 'L6a': 'L6', 'L6b':'L6', 'L6': 'L6', 'nan' : 'nan'}

def instantiate_KG():
    
    G = nx.Graph()
    G.add_node('efeature', color='orange')
    G.add_node('mfeature', color='g')
    G.add_node('No_ephys_data', color='brown')
    G.add_node('No_morpho_data', color='darkgreen')

    G.add_node('molecularID', color='darkblue')
        
    G.add_node('No_labels', color='red')
    
    return G


def add_dataset_to_KG(G, me_data, labels, dataset_name, color_cell, color_label,
                      label_name_list=["me-type", "m-type", "e-type", "layer", "marker"]):

    G.add_node(dataset_name + "_labels", color=color_label)
    for lbl_name in label_name_list:
        if lbl_name in labels.columns:
            for sub_label in unique_elements(labels[lbl_name]):
                if lbl_name != "marker":
                    if lbl_name == "layer":
                        G.add_node(dict_layer[sub_label], color="purple")
                        G.add_edge(dataset_name + "_labels", dict_layer[sub_label], color='silver')
                    else:
                        G.add_node(sub_label, color=color_label)
                        G.add_edge(dataset_name + "_labels", sub_label, color='silver')
                elif lbl_name == "marker":
                    G.add_node(sub_label, color="darkblue")
                    G.add_edge(dataset_name + "_labels", sub_label, color='silver')
                    G.add_edge("molecularID", sub_label, color='silver')

    for cell in me_data.index:

        G.add_node(cell, color=color_cell)
        try:
            metype = labels["me-type"][cell]
            mtype = labels["m-type"][cell]
            etype = labels["e-type"][cell]
            lay = dict_layer[labels["layer"][cell]]
            G.add_edge(cell, metype, color='silver')
            G.add_edge(cell, mtype, color='silver')
            G.add_edge(cell, etype, color='silver')
            G.add_edge(cell, lay, color='silver')
            if "marker" in labels.columns:
                marker = labels["marker"][cell]
                G.add_edge(cell, marker, color='silver')
                G.add_edge("molecularID", marker, color='silver')

        except:
            G.add_edge(cell, "No_labels", color='silver')
            print("No labels for cell " + dataset_name , str(cell))


        msk_morpho = np.asarray(["morpho" in col for col in me_data.columns])
        e_features_list = me_data.columns[~msk_morpho]
        m_features_list = me_data.columns[msk_morpho]

        for efeat in e_features_list:
            try:
                if str(me_data[efeat][cell]) != 'nan':
                    G.add_node(efeat, color='gold')
                    G.add_edge(cell, efeat, color='silver')
                    G.add_edge('efeature', efeat, color='yellow')
            except:
                G.add_edge(cell, 'No_ephys_data', color='silver')

        for mfeat in m_features_list:
            try:
                if str(me_data[mfeat][cell]) != 'nan':
                    G.add_node(mfeat, color='g')
                    G.add_edge(cell, mfeat, color='silver')
                    G.add_edge('mfeature', mfeat, color='lightgreen')
            except:
                G.add_edge(cell, 'No_morpho_data', color='silver')

    return G

    
def plot_KG(G, show_plot=True):

    from matplotlib.lines import Line2D

    legend_elements = []

    for c,lbl in zip(['blue', 'darkblue', 'lightblue', 'purple', 'k', 'grey', 'g', 'darkgreen',
                      'gold', 'brown', 'red'],
                     ['Gouwens labels', 'mol_ID', 'BBP labels', 'layer', 'AIBS_cells',
                      'BBP_cells', 'm_features', 'no morpho', 'e_features', 'no ephys', 'No_label']):
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=lbl,
                                      markerfacecolor=c, markersize=10))

    for c,lbl in zip(['silver', 'lightgreen', 'yellow'],
                     ['Cell_link', 'morpho', 'ephys']):
        
        legend_elements.append(Line2D([0], [0], color=c, lw=1, label=lbl))

    options = {
         'node_color': [G.nodes[x]['color'] for x in G.nodes],
         'node_size': 20,
         'width': 0.5,
         'edge_color' : [G.edges[x]['color'] for x in G.edges]}
    plt.figure(figsize=(10,10))
    plt.subplot(111)
    nx.draw(G, with_labels=False, **options)
    plt.legend(handles=legend_elements)
    if show_plot:
        plt.show()

    if not os.path.isdir("./figures"):
        os.mkdir("./figures")
    plt.savefig('./figures/knowledge_graph.pdf', format='pdf')

    return

def KG_analysis_shared_features(G, list_id_list, dset_names, threshold_dset_name,
                                threshold=0.):
    KG_dset_features = []
    for id_list in list_id_list:
        KG_features = []
        for cell in id_list:
            KG_features+=[x for x in nx.all_neighbors(G,cell)]
        counts = count_elements(np.asarray(KG_features))
        KG_dset_features.append(counts)
    
    kept_features = pd.concat([KG_dset_features[i].rename({'counts':dset_names[i]}, axis=1)
                               for i in range(len(dset_names))],
                              axis=1)

    kept_features = kept_features[[n>threshold for n in kept_features[threshold_dset_name].values]]
    kept_features = kept_features.sort_values(by=[threshold_dset_name], ascending=False)
    
    return kept_features


def build_dataset(me_df, step_list, id_list, efeat_noprot_list, mfeat_list):
    features = []

    for cell in id_list:

        feature_vals = []
        for ef in efeat_noprot_list:

            check_val = False
            i = 0
            while check_val == False:

                protocol = step_list[i]
                ef_name = ef + '|' + protocol

                ef_val = me_df[ef_name][cell]

                if (type(ef_val) == np.float64) & np.isfinite(ef_val):
                    feature_vals.append(ef_val)
                    check_val = True

                else:
                    if protocol != step_list[-1]:
                        i += 1

                    elif protocol == step_list[-1]:
                        feature_vals.append(np.nan)
                        check_val = True

        try:
            for mf in mfeat_list:
                feature_vals.append(me_df[mf][cell])
        except:
            feature_vals = feature_vals + [np.nan] * len(mfeat_list)

        features.append(np.asarray(feature_vals))

    dataset = pd.DataFrame(np.asarray(features), index=id_list, columns=efeat_noprot_list + mfeat_list)

    return dataset

def ref_feature_coverage(dataset_dict):
    
    vals = {}
    
    for dset_name in dataset_dict.keys():
        dset = dataset_dict[dset_name]
        
        vals[dset_name] = ((np.asarray([np.sum(np.isfinite(dset[col].values)) for col in dset.columns])
                            / len(dset))
                          )

    feature_coverage_ref = pd.concat([pd.DataFrame(vals[dset_name], 
                                                   index=dataset_dict[dset_name].columns, 
                                                   columns=[dset_name + ' cell values %'])
                                      for dset_name in dataset_dict.keys()],
                                 axis=1)

    return feature_coverage_ref
    
def release_incomplete_samples(selected_features, dataset):
    
    incomplete_samples= []
    
    for col in selected_features:
        candidates = dataset.index[~np.isfinite(dataset[col].values)]

        for x in candidates:
            if x not in incomplete_samples:
                incomplete_samples.append(x)
    
    return incomplete_samples

def shared_features(th, feature_coverage_ref, dataset_dict):

    msk_features = (feature_coverage_ref>th).product(axis=1).values
    msk_features = np.ma.make_mask(msk_features)

    selected_features = feature_coverage_ref.index[msk_features]
    
    incomplete_samples_dict = {}
    
    for dset_name in dataset_dict.keys():
        
        incomplete_samples_dict[dset_name] = release_incomplete_samples(selected_features, dataset_dict[dset_name])

    return selected_features, incomplete_samples_dict

def Datasets_analysis_shared_features(threshold_values, feature_coverage_ref, 
                                      dataset_dict, list_id_list):

    features_coverage = []
    datasets_coverage = {dset_name : [] for dset_name in dataset_dict.keys()}

    for th in threshold_values:

        selected_features, incomplete_samples_dict = shared_features(th, feature_coverage_ref, dataset_dict)

        features_coverage.append(len(selected_features) / len(feature_coverage_ref.index))
        for dset_name, id_list in zip(dataset_dict.keys(), list_id_list):
            datasets_coverage[dset_name].append(len(incomplete_samples_dict[dset_name]) / len(id_list))

    plt.figure(figsize=(4, 3))
    plt.subplot(111)
    plt.plot(threshold_values, features_coverage)
    for dset_name in datasets_coverage.keys():
        plt.plot(threshold_values, 1 - np.asarray(datasets_coverage[dset_name]))
    plt.legend(['features'] + [k for k in datasets_coverage.keys()])
    plt.xlabel('threshold value')
    plt.ylabel('coverage ratio')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/dataset_coverageVSthreshold.pdf', format='pdf')
    
    return features_coverage, datasets_coverage