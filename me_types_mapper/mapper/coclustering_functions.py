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

import scipy
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import cluster
from scipy.cluster.hierarchy import fcluster

# define custom functions

def unique_elements(array):
    """Return unique elements of an array."""
    
    unique = []
    for x in array:
        if x not in unique:
            unique.append(x)
    return unique

def count_elements(array):
    """Return as and pandas DataFrame unique elements and the associated counts of an array."""
    
    unq = unique_elements(array)
    
    return pd.DataFrame([len(array[[y==x for y in array]]) for x in unq], index=unq, columns=['counts'])

def derivative(x,y):
    """take the derivative of an array of values"""
    deriv_ = []
    
    for i in range(len(y)-1):
        deriv_.append((y[i+1] - y[i]) / (x[i+1] - x[i]))
    
    return x[1:], np.asarray(deriv_)

def smooth(y, box_pts):
    """Convolve an array with a step function of a given width. Results in a smoothed signal."""

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

###Preprocessing datasets functions ###

def dataset_clearing(X, msk_ephys_, msk_morpho_):
    """Remove sample with NaNs for all electrophysiological features or all morphological features."""

    e_idx = X[X.columns[msk_ephys_]].dropna(axis=0, how='all').index 
    m_idx = X[X.columns[msk_morpho_]].dropna(axis=0, how='all').index

    common_idx = np.asarray(list(set(e_idx).intersection(m_idx)))
    
    return X.T[common_idx].T

def Split_raw(X_data, msk_ephys_, msk_morpho_):
    """Split dataset into two tables. One for efeatures only and one for mfeatures only."""
    
    X_ephys_data = pd.DataFrame(X_data[X_data.columns[msk_ephys_]].fillna(0).values,
                                index=X_data.index,
                                columns=X_data.columns[msk_ephys_])
    
    
    X_morpho_data = pd.DataFrame(X_data[X_data.columns[msk_morpho_]].fillna(0).values,
                                 index=X_data.index,
                                 columns=X_data.columns[msk_morpho_])
    return X_ephys_data, X_morpho_data

def Z_scale(X_data, msk_ephys_, msk_morpho_):
    """Z-scaling (i.e. set mean to 0 and var to 1 for each individual e-feature an m-features)"""

    from sklearn import preprocessing
    
    X_ephys_data = pd.DataFrame(preprocessing.scale(X_data[X_data.columns[msk_ephys_]].fillna(0).values),
                                index=X_data.index,
                                columns=X_data.columns[msk_ephys_])
    
    
    X_morpho_data = pd.DataFrame(preprocessing.scale(X_data[X_data.columns[msk_morpho_]].fillna(0).values),
                                 index=X_data.index,
                                 columns=X_data.columns[msk_morpho_])
    return X_ephys_data, X_morpho_data

def PCA_projection(X_ref, X_test):
    """Proceed to PCA on reference dataset (X_ref) and apply the same transformation on test dataset (X_test)."""

    from sklearn.decomposition import PCA

    pca = PCA(n_components=np.min([len(X_ref.columns), len(X_ref), len(X_test.columns), len(X_test)]))

    X_ref_pca = pd.DataFrame(pca.fit_transform(X_ref.values), index=X_ref.index)

    X_test_pca = pd.DataFrame(pca.transform(X_test.values), index=X_test.index)

    return pd.concat([X_ref_pca,X_test_pca], axis=0)

def Combine_data(X_ref, X_test):
    """merge both dataset."""

    from sklearn.decomposition import PCA

    X_ = pd.concat([X_ref, X_test], axis=0)
    pca_ = PCA(n_components=np.min([len(X_.columns), len(X_)]))

    return pd.DataFrame(pca_.fit_transform(X_.values), index=X_.index)

def Combine_me_data(Xe_ref, Xe_test, Xm_ref, Xm_test, alpha):
    """merge both dataset with an alpha weight between e-features and m-features."""

    from sklearn.decomposition import PCA
    
    Xe_ = pd.concat([Xe_ref, Xe_test], axis=0)
    Xm_ = pd.concat([Xm_ref, Xm_test], axis=0)
    Xme_ = pd.concat([(1-alpha)*Xe_, alpha*Xm_], axis=1)
    
    
    pca_ = PCA(n_components=np.min([len(Xme_.columns), len(Xme_)]))
    Xme_pca = pca_.fit_transform(Xme_.values)
    
    var_r = pca_.explained_variance_ratio_
    cum_var_r = np.asarray([np.sum(var_r[:i])for i in range(len(var_r))])
    n_pc = np.where(cum_var_r>0.99)[0][0]
    
    return pd.DataFrame(Xme_pca[:,:n_pc], index=Xme_.index)

def preprocess_data(data_ref, data_test, msk_ephys_, msk_morpho_, alpha):
    """Preprocessing step for a dataset containing Z-scoring and PCA."""
    
    # Z_scored
    Xe_ref, Xm_ref = Z_scale(data_ref, msk_ephys_, msk_morpho_)
    Xe_test, Xm_test = Z_scale(data_test, msk_ephys_, msk_morpho_)
    
    #Concatenation + PCA
    
    Xme = Combine_me_data(Xe_ref, Xe_test, Xm_ref, Xm_test, alpha)
    
    return Xme
    

### alpha opt functions ###

def name_clusters_ref(leave_labels, clusters_ref, msk_):
    clusters_ref_names = []

    for clstr in unique_elements(clusters_ref):
        msk_c = [c==clstr for c in clusters_ref[msk_]]
        cts_ = count_elements(np.asarray(leave_labels)[msk_][msk_c])

        try:
            clusters_ref_names.append('|'.join(cts_.index[np.where(cts_.values == np.max(cts_.values))[0]]))
        except:
            clusters_ref_names.append('No_match')
            
    order = np.argsort(unique_elements(clusters_ref))     
    dict_clusters_ref = {np.asarray(unique_elements(clusters_ref))[order][i] : np.asarray(clusters_ref_names)[order][i] 
                         for i in range(len(clusters_ref_names))}
    
    return [dict_clusters_ref[lbl] for lbl in clusters_ref]

def percentage_label_cluster(Z, dist_array, labels, msk_set = None):
    
    if msk_set == None:
        msk_set = [True]*len(labels)
    
    percentage_mean = []
    percentage_std = []
    
    for max_d in dist_array:
        clusters_ref = fcluster(Z, max_d, criterion='distance')

        percentage = []
        
        for clstr in unique_elements(clusters_ref[msk_set]):
            msk_ = [c == clstr for c in clusters_ref[msk_set]]

            percentage.append(np.max(count_elements(labels[msk_set][msk_]) / len(labels[msk_set][msk_]))[0])
            
        percentage_mean.append(np.mean(np.asarray(percentage)))
        percentage_std.append(np.std(np.asarray(percentage)))

    return np.asarray(percentage_mean), np.asarray(percentage_std)


def alpha_opt_v3(data_ref, data_test, msk_ephys_, msk_morpho_, 
                 coeff_list, labels, msk_set= None, hca_method='ward', plotting=True, cmap_color='Blues'):
    
    from sklearn.metrics import auc
    cmap = matplotlib.cm.get_cmap(cmap_color)
    
    dist_array = np.arange(1,101,1)
    percentage_labels_mean_list = []
    
    for coeff_ in coeff_list:
        
        X_transformed_df = preprocess_data(data_ref, data_test, msk_ephys_, msk_morpho_, coeff_)
        
        distances = scipy.spatial.distance.pdist(X_transformed_df,metric='euclidean')
        
        Z = cluster.hierarchy.linkage(distances, method=hca_method, metric='euclidean', optimal_ordering=False)
    
        dist_array_ = dist_array*np.max(Z[:,2])*0.01
        
        percentage_labels_mean = percentage_label_cluster(Z, dist_array_, labels, msk_set)
    
        percentage_labels_mean_list.append(percentage_labels_mean)
        
    percentage_labels_mean_list = np.asarray(percentage_labels_mean_list)
    
#     integral_vals = [np.mean(p[0]) for p in percentage_labels_mean_list]
    integral_vals = [auc(dist_array*0.01,p[0]) for p in percentage_labels_mean_list]
    alpha_opt_ = coeff_list[np.where(integral_vals==np.max(integral_vals))[0][0]]
    
    ##plotting
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=(8, 2.2))
    if plotting == True:

        for ax_ in ax.flatten():
            right_side = ax_.spines["right"]
            top_side = ax_.spines["top"]
            right_side.set_visible(False)
            top_side.set_visible(False)

        for i,percentage_labels_mean in enumerate(percentage_labels_mean_list):
            ax[0].plot(dist_array, percentage_labels_mean[0], 
                    c=cmap((i+1)/len(percentage_labels_mean_list)))
        # ax[0].set_xlabel('Normalised clustering distance %', fontsize=14)
        # ax[0].set_title('Average cluster homogeneity', fontsize=16)
        ax[0].set_ylim([0.,1.])
        ax[0].set_xlim([0.,100.])
        ax[0].set_xticks([0., 50, 100])
        ax[0].set_xticklabels([0, 50, 100], fontsize=14)
        ax[0].set_yticks([0., .5, 1.])
        ax[0].set_yticklabels([0, .5, 1], fontsize=14)

        ax[1].plot(coeff_list, integral_vals, '-o', c='k', ms=8)
        for i, (x_, y_) in enumerate(zip(coeff_list, integral_vals)):
            ax[1].plot(x_, y_, 'o', c=cmap((i+1)/len(coeff_list)), ms=7, clip_on=False)
        # ax[1].set_xlabel('α coefficient', fontsize=14)
        # ax[1].set_title('Integral value', fontsize=16)
        ax[1].axvline(x=alpha_opt_, c='k', ls='--')
        ax[1].text(alpha_opt_,0.95, 'α = ' + str(round(alpha_opt_, 2)), fontsize=16)
        ax[1].set_ylim([0., 2.])
        ax[1].set_ylim([0., 1.])
        ax[1].set_xticks([0., .5, 1.])
        # ax[1].set_xticks([0, 1, 2, 3, 4, 5])
        ax[1].set_xticklabels([0, 0.5, 1], fontsize=14)
        ax[1].set_yticks([0., .5, 1.])
        ax[1].set_yticklabels([0, .5, 1], fontsize=14)
        #     ax[1].set_xlim([1.,1.5])
    #     plt.show()
    #     plt.savefig("./alpha_optimization.pdf", format="pdf")

    return percentage_labels_mean_list, integral_vals, alpha_opt_, fig

### clustering opt ###

def prepare_cocluster_data(X_1_df, X_2_df, axis=0):
    
    from sklearn.decomposition import PCA
    
    X_cocluster_df = pd.concat([X_1_df, X_2_df], axis=axis)
    
    transformer = PCA(n_components=np.min([len(X_cocluster_df.columns), len(X_cocluster_df)]))
    
    
    X_cocluster_transformed = transformer.fit_transform(X_cocluster_df.values)
    X_cocluster_transformed_df = pd.DataFrame(X_cocluster_transformed, index=X_cocluster_df.index,
                                              columns=['PC_' + str(i) for i in range(len(X_cocluster_transformed[0,:]))])
    
    return X_cocluster_transformed_df
    
def find_label_agreement(X_1_df, X_2_df, labels, hca_method='ward'):
    
    dist_array = np.arange(1,101,1)
    
    X_cocluster_transformed_df = prepare_cocluster_data(X_1_df, X_2_df)
    
    X2_No_match = []
    X1_No_match = []
    agree = []
    disagree = []

    
    distances = scipy.spatial.distance.pdist(X_cocluster_transformed_df,metric='euclidean')
    Z = cluster.hierarchy.linkage(distances, method=hca_method, metric='euclidean', optimal_ordering=False)
    dist_array_ = dist_array*np.max(Z[:,2])*0.01
    
    msk_1 = [True] * len(X_1_df.index) + [False] * len(X_2_df.index)
    msk_2 = [False] * len(X_1_df.index) + [True] * len(X_2_df.index)

    for max_d in dist_array_:
        clusters_ref = fcluster(Z, max_d, criterion='distance')

        clusters_ref_names_1 = np.asarray(name_clusters_ref(labels, clusters_ref, msk_1))
        clusters_ref_names_2 = np.asarray(name_clusters_ref(labels, clusters_ref, msk_2))

        counts_X2_No_match = 0.
        counts_X1_No_match = 0.
        counts_agree = 0.
        counts_disagree = 0.

        for i,lbl in enumerate(clusters_ref_names_2):

            if lbl == 'No_match':
                counts_X2_No_match +=1

            elif clusters_ref_names_1[i] == 'No_match':
                counts_X1_No_match +=1

            elif lbl in clusters_ref_names_1[i]:
                counts_agree +=1

            elif lbl not in clusters_ref_names_1[i]:
                counts_disagree +=1

        X2_No_match.append(counts_X2_No_match)
        X1_No_match.append(counts_X1_No_match)
        agree.append(counts_agree)
        disagree.append(counts_disagree)

    X2_No_match = np.asarray(X2_No_match) * 100. / len(np.asarray(labels))
    X1_No_match = np.asarray(X1_No_match) * 100. / len(np.asarray(labels))
    agree = np.asarray(agree) * 100. / len(np.asarray(labels))
    disagree = np.asarray(disagree) * 100. / len(np.asarray(labels))
    
    return X1_No_match, X2_No_match, agree, disagree

def find_clustering_distance(X_1_df, X_2_df, labels, Z, msk_set=None, hca_method='ward'):
    
    dist_array = np.arange(1,101,1)
    dist_array_ = np.arange(.01,1.01,.01)*np.max(Z[:,2])
        
    percentage_labels_mean = percentage_label_cluster(Z, dist_array_, labels, msk_set=msk_set)
    
    X1_No_match, X2_No_match, agree, disagree = find_label_agreement(X_1_df, X_2_df, labels, hca_method=hca_method)


    idx_ = np.where((agree + disagree) > (percentage_labels_mean[0] * 100))
    
    d_opt = dist_array[idx_][0]
    
    ##plotting##
    fig, ax = plt.subplots(figsize=(4,2.2))
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)

    ax.plot(dist_array,  agree + disagree, c='red', alpha=0.7)
    ax.plot(dist_array, percentage_labels_mean[0] * 100, 
            c='b', alpha=0.7)
    ax.fill_between(dist_array, 
                    (percentage_labels_mean[0] - percentage_labels_mean[1]) * 100,
                    (percentage_labels_mean[0] + percentage_labels_mean[1]) * 100,
                    color='b', alpha=0.2)
    ax.set_xticks([0., 50., 100.])
    ax.set_xticklabels([0, 50, 100], fontsize=14)
    ax.set_yticks([0., 50., 100.])
    ax.set_yticklabels([0, 50, 100], fontsize=14)
    
    plt.axvline(x=dist_array[idx_][0], ls='--', c='k')
    plt.ylim([0,101.])
    plt.xlim([0,100.])
    plt.legend(['match %','Homogeneity'], loc='lower right', fontsize=12)
    fig.text(0.15, 0.2, 'd_opt = ' + str(dist_array[idx_][0]), color='k', fontsize=14)
    fig.text(dist_array[idx_][0]*.01 + .06, 0.58, '%.2f' % (percentage_labels_mean[0] * 100)[idx_][0], color='blue', fontsize=14)
    fig.text(dist_array[idx_][0]*.01 + .06, 0.81, '%.2f' % (agree + disagree)[idx_][0], color='red', fontsize=14)
    # plt.show()
    # plt.savefig("./clustering_distance_optimization.pdf", format="pdf")
    
    return d_opt, fig

def forming_clusters(X_1_df, X_2_df, labels, msk_set = None, d_opt = None, hca_method='ward'):
    
    X_cocluster_transformed_df = prepare_cocluster_data(X_1_df, X_2_df)
    
    distances = scipy.spatial.distance.pdist(X_cocluster_transformed_df, metric='euclidean')
    
    Z = cluster.hierarchy.linkage(distances, method=hca_method, metric='euclidean', optimal_ordering=False)
    if d_opt == None:
        d_opt, fig_d_opt = find_clustering_distance(X_1_df, X_2_df, labels, Z, msk_set = msk_set, hca_method=hca_method)

    print('d_opt = ', d_opt)
    max_d = .01*d_opt*np.max(Z[:,2])
    clusters_ref = fcluster(Z, max_d, criterion='distance')
    
    return clusters_ref, fig_d_opt

def mapping_cluster_ref(labels, cluster_ref):
    
    counts_frames = []

    for xtype in unique_elements(np.asarray(labels)):
        msk_ = [lbl == xtype for lbl in np.asarray(labels)]
        counts_frames.append(count_elements(np.asarray(cluster_ref)[msk_]).reindex(unique_elements(cluster_ref)).rename(columns={'counts':xtype}))

    clusters_labels1_map = pd.concat(counts_frames, axis=1)
    clusters_labels1_map = clusters_labels1_map.reindex(clusters_labels1_map.index[np.argsort(clusters_labels1_map.index)])
    clusters_labels1_map = clusters_labels1_map.reindex(columns= clusters_labels1_map.columns[np.argsort(clusters_labels1_map.columns)])
    clusters_labels1_map = clusters_labels1_map.fillna(0)
#     clusters_labels1_map = clusters_labels1_map.div(np.sum(clusters_labels1_map, axis=1),axis=0)

    return clusters_labels1_map

def mapping(labels1, labels2, cluster_ref):
    
    clusters_labels1_map = mapping_cluster_ref(labels1, cluster_ref)
    clusters_labels2_map = mapping_cluster_ref(labels2, cluster_ref)
    map_ = clusters_labels1_map.T @ clusters_labels2_map.reindex(clusters_labels1_map.index).fillna(0)
    
    map__ = map_.drop('no_label').T.drop('no_label').T
    
    c1_ = clusters_labels1_map.T.drop('no_label').T
    
    dict_cluster_label = {}
    for idx in c1_.index:
        label_idx = np.where(c1_.T[idx]==np.max(c1_.T[idx]))[0]
        if len(label_idx)==1:
            dict_cluster_label[idx] = c1_.columns[label_idx[0]]
        else:
            dict_cluster_label[idx] = 'no_prediction'
    
#     plt.figure(figsize=(6,6))
#     plt.imshow(map__.fillna(0.0).values, cmap='jet')
#     plt.xticks(np.arange(len(map__.columns)), map__.columns, rotation=90, ha='center')
#     plt.yticks(np.arange(len(map__.index)), map__.index)
# #     plt.colorbar(shrink=0.9)
#     plt.ylabel('train')
#     plt.xlabel('test')
#     plt.show()
#     plt.savefig
    
    return (map__,clusters_labels1_map, clusters_labels2_map, dict_cluster_label)


def cross_predictions_v2(data_1, data_2, msk_ephys_, msk_morpho_, lbls,
                      alpha_list_ = np.arange(.1,1.,.1), d_opt=None, hca_method='ward'):
    
    msk_1 = [True]*len(data_1) + [False]*len(data_2)
    msk_2 = [False]*len(data_1) + [True]*len(data_2)

    label_list = np.asarray(lbls[msk_1].tolist() + [None]*len(lbls[msk_2]))


    percentage_labels_mean_list, integral_vals, alpha_opt, fig_alpha = alpha_opt_v3(data_1, data_2,
                                                                                    msk_ephys_, msk_morpho_,
                                                                                    alpha_list_, label_list,
                                                                                    msk_set=msk_1,
                                                                                    hca_method='ward',
                                                                                    cmap_color='Reds')
    
    X_df = preprocess_data(data_1, data_2, msk_ephys_, msk_morpho_, alpha_opt)

    X1_df = X_df[msk_1]
    X2_df = X_df[msk_2]
   
    print('alpha = '+ str(round(alpha_opt,2)))

    label_lst = np.asarray(lbls[msk_1].tolist() + ['no_label']*len(lbls[msk_2]))
    label_lst2 = np.asarray(['no_label']*len(lbls[msk_1]) + lbls[msk_2].tolist())

    cluster_ref, fig_d_opt = forming_clusters(X1_df, X2_df, labels=label_lst, msk_set=msk_1,
                                              d_opt=d_opt, hca_method=hca_method)

    map_, c1, c2, dict_cluster_label = mapping(label_lst, label_lst2, cluster_ref)
    
    return alpha_opt, map_, c1, c2, dict_cluster_label, cluster_ref, fig_alpha, fig_d_opt

def compute_probabilistic_maps(labels_test_list, label_test_name, mask_test,
                               labels_ref_list, label_ref_name, mask_ref,
                               cluster_reference_dict):
    """
    Exploit the native labels from test and reference dataset in addition to their common cluster labels to compute
    the probability matrices:
    P(label_test|label_ref) = P(lbl_test|c).T @ P(c|lbl_ref)
    P(label_ref|label_test) = P(lbl_ref|c).T @ P(c|lbl_test)
    Args:
        labels_test_list:
        label_test_name:
        mask_test:
        labels_ref_list:
        label_ref_name:
        mask_ref:
        cluster_reference_dict:

    Returns:

    """
    cell_ids = np.asarray([x for x in cluster_reference_dict.keys()])
    clstr_list = np.asarray([x for x in cluster_reference_dict.values()])

    lbls_test = np.asarray([labels_test_list[label_test_name][x] for x in cell_ids[mask_test]])
    c_test = mapping_cluster_ref(lbls_test, clstr_list[mask_test]).reindex(unique_elements(clstr_list)).fillna(0.)
    P_lbl_test_c = c_test.div(np.sum(c_test, axis=1), axis=0).reindex(unique_elements([x for x in labels_test_list[label_test_name]]),
                                                                      axis=1).fillna(0.)  # P(lbl|c)
    p_c_lbl_test = (c_test.div(count_elements(lbls_test)['counts'], axis=1)).reindex(
        unique_elements([x for x in labels_test_list[label_test_name]]), axis=1).fillna(0.)  # P(c|lbl)

    lbls_ref = np.asarray([labels_ref_list[label_ref_name][np.float(x)] for x in cell_ids[mask_ref]])
    c_ref = mapping_cluster_ref(lbls_ref, clstr_list[mask_ref]).reindex(unique_elements(clstr_list)).fillna(0.)
    P_lbl_ref_c = c_ref.div(np.sum(c_ref, axis=1), axis=0).reindex(
        unique_elements([x for x in labels_ref_list[label_ref_name]]), axis=1).fillna(0.)  # P(lbl|c)
    P_c_lbl_ref = (c_ref.div(count_elements(lbls_ref)['counts'], axis=1)).reindex(
        unique_elements([x for x in labels_ref_list[label_ref_name]]), axis=1).fillna(0.)  # P(c|lbl)

    clstr_order = np.sort(P_lbl_test_c.index)
    P_lbl_test_c = P_lbl_test_c.reindex(clstr_order)
    P_c_lbl_ref = P_c_lbl_ref.reindex(clstr_order)
    P_lbl_ref_c = P_lbl_ref_c.reindex(clstr_order)
    p_c_lbl_test = p_c_lbl_test.reindex(clstr_order)

    P_label_test_label_ref = P_lbl_test_c.T @ P_c_lbl_ref
    P_label_ref_label_test = P_lbl_ref_c.T @ p_c_lbl_test

    ref_lbl_order = np.sort(unique_elements(labels_ref_list[label_ref_name]))
    P_label_test_label_ref = P_label_test_label_ref.reindex(ref_lbl_order, axis=1)
    P_c_lbl_ref = P_c_lbl_ref.reindex(ref_lbl_order, axis=1)
    P_label_ref_label_test = P_label_ref_label_test.reindex(ref_lbl_order, axis=0)

    return c_ref, P_lbl_ref_c, P_c_lbl_ref, \
           c_test, P_lbl_test_c, p_c_lbl_test, \
           P_label_test_label_ref, P_label_ref_label_test