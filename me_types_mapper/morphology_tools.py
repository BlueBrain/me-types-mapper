""" Functions to normalize morphologies and extract density moments for both AIBS and BBP morphologies """
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

import pandas as pd
import numpy as np
import neurom as nm
import os

from morphio.mut import Morphology

def unique_elements(array):
    unique = []
    for x in array:
        if x not in unique:
            unique.append(x)
    return unique

def count_elements(array):
    unq = unique_elements(array)

    return pd.DataFrame([len(array[[y==x for y in array]]) for x in unq], index=unq, columns=['counts'])

cortical_layer_thickness = {}
cortical_layer_thickness['Rat'] = {'S1': {'Total': 1827, 'pia': 0.,'L1': 0.068*1827, 'L23': (0.068+0.254)*1827, 'L4': (0.068+0.254+0.085)*1827, 'L5': (0.068+0.254+0.085+0.297)*1827, 'L6': 1827}}
cortical_layer_thickness['Mouse'] = {'S1': {'Total': 1718, 'L1': 0.06248*1718, 'L23': (0.06248+0.2302)*1718, 'L4': (0.06248+0.2302+0.1339)*1718, 'L5': (0.06248+0.2302+0.1339+0.2632)*1718, 'L6': 1718},
                                     'V1': {'Total': 1382, 'L1': 0.08507*1382, 'L23': (0.08507+0.2441)*1382, 'L4': (0.08507+0.2441+0.1316)*1382, 'L5': (0.08507+0.2441+0.1316+0.2416)*1382, 'L6': 1382},
                                     'MOp': {'L1':0.07, 'L23':0.29, 'L5': 0.73}
}

def y_layer_percent(y, animal, region):
    if y > -cortical_layer_thickness[animal][region]['L1']:
        y_new = y/cortical_layer_thickness[animal][region]['L1']
    elif (y < -cortical_layer_thickness[animal][region]['L1'])&(y > -cortical_layer_thickness[animal][region]['L23']):
        y_new = -1 + (y+cortical_layer_thickness[animal][region]['L1'])/(cortical_layer_thickness[animal][region]['L23']-cortical_layer_thickness[animal][region]['L1'])
    elif (y < -cortical_layer_thickness[animal][region]['L23'])&(y > -cortical_layer_thickness[animal][region]['L4']):
        y_new = -2 + (y+cortical_layer_thickness[animal][region]['L23'])/(cortical_layer_thickness[animal][region]['L4']-cortical_layer_thickness[animal][region]['L23'])
    elif (y < -cortical_layer_thickness[animal][region]['L4'])&(y > -cortical_layer_thickness[animal][region]['L5']):
        y_new = -3 + (y+cortical_layer_thickness[animal][region]['L4'])/(cortical_layer_thickness[animal][region]['L5']-cortical_layer_thickness[animal][region]['L4'])
    elif (y < -cortical_layer_thickness[animal][region]['L5']):#&(y > -cortical_layer_thickness[animal][region]['L6']):
        y_new = -4 + (y+cortical_layer_thickness[animal][region]['L5'])/(cortical_layer_thickness[animal][region]['L6']-cortical_layer_thickness[animal][region]['L5'])
    else:
        y_new = y
    return y_new


def rotateXYcoords(coords, theta):
    R = np.asarray([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return R @ coords.transpose()

def rotateXYZcoords(coords, theta):
    R = np.asarray([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.],
                    [ 0., 0., 1.,]])
    return R @ coords.transpose()


def arbor_segments(morpho):
    segments_ax = []
    segments_dend = []

    for section in morpho.iter():
        if "axon" in str(section.type):
            for i in range(len(section.points) - 1):
                x_midpoint = np.mean(section.points[i][0] + section.points[i + 1][0])
                y_midpoint = np.mean(section.points[i][1] + section.points[i + 1][1])
                len_seg = np.sqrt((section.points[i + 1][0] - section.points[i][0]) ** 2 +
                                  (section.points[i + 1][1] - section.points[i][1]) ** 2)

                segments_ax.append([x_midpoint, y_midpoint, len_seg])

            for child in section.children:
                s_xy = section.points[-1]
                c_xy = child.points[0]

                x_midpoint = np.mean(s_xy[0] + c_xy[0])
                y_midpoint = np.mean(s_xy[1] + c_xy[1])
                len_seg = np.sqrt((c_xy[0] - s_xy[0]) ** 2 + (c_xy[1] - s_xy[1]) ** 2)

                segments_ax.append([x_midpoint, y_midpoint, len_seg])

        if "dendrite" in str(section.type):
            for i in range(len(section.points) - 1):
                x_midpoint = np.mean(section.points[i][0] + section.points[i + 1][0])
                y_midpoint = np.mean(section.points[i][1] + section.points[i + 1][1])
                len_seg = np.sqrt((section.points[i + 1][0] - section.points[i][0]) ** 2 +
                                  (section.points[i + 1][1] - section.points[i][1]) ** 2)

                segments_dend.append([x_midpoint, y_midpoint, len_seg])

            for child in section.children:
                s_xy = section.points[-1]
                c_xy = child.points[0]

                x_midpoint = np.mean(s_xy[0] + c_xy[0])
                y_midpoint = np.mean(s_xy[1] + c_xy[1])
                len_seg = np.sqrt((c_xy[0] - s_xy[0]) ** 2 + (c_xy[1] - s_xy[1]) ** 2)

                segments_dend.append([x_midpoint, y_midpoint, len_seg])

    l_max_dend = np.sum(np.asarray(segments_dend)[:, 2])
    l_max_ax = np.sum(np.asarray(segments_ax)[:, 2])

    segments_f_ax = np.asarray([[seg[0],
                                 seg[1],
                                 seg[2] / l_max_ax] for seg in segments_ax])
    segments_f_dend = np.asarray([[seg[0],
                                   seg[1],
                                   seg[2] / l_max_dend] for seg in segments_dend])

    # fig, ax = plt.subplots()
    # ax.scatter(np.asarray(segments_f_ax)[:, 0], np.asarray(segments_f_ax)[:, 1], s=1, alpha=.1)
    # ax.scatter(np.asarray(segments_f_dend)[:, 0], np.asarray(segments_f_dend)[:, 1], s=1, alpha=.1)
    # ax.legend(['axon', 'dendrite'])

    return np.unique(segments_f_ax, axis=0), np.unique(segments_f_dend, axis=0)


def moment_j_k(segments,j,k):
    return np.sum(np.asarray(segments)[:,0]**j * np.asarray(segments)[:,1]**k * np.asarray(segments)[:,2])

def compute_moments(segments, N=5):
    
    moments = []
    
    try:
        for k in range(N):

            m0 = np.sum(segments[:,2])

            moments_xk = moment_j_k(segments,k,0) / (np.sqrt(m0))
            moments_yk = moment_j_k(segments,0,k) / (np.sqrt(m0))
            moments_k = moment_j_k(segments,k,k)

            moments.append([moments_xk, moments_yk, moments_k])
    except:
        moments.append(['No_reconstruction'])
        
    return np.asarray(moments)


def normalize_morphology(path_to_morphology, distance_to_the_pia, layer, layer_sup, animal, brain_region,
                         rotation_angle=0., realign=True):
    m = Morphology(path_to_morphology)
    region = brain_region

    if distance_to_the_pia != None:
        y_soma = distance_to_the_pia


    elif (layer, layer_sup) != (None, None):
        y_lay_sup = cortical_layer_thickness[animal][region][layer_sup]
        y_lay = cortical_layer_thickness[animal][region][layer]
        y_soma = 0.5 * (y_lay_sup + y_lay)

    y_list = []

    for s in m.iter():
        s.points = s.points - m.soma.points[0]
        if rotation_angle != 0.:
            s.points = np.asarray([rotateXYZcoords(pts, rotation_angle)
                                   for pts in s.points
                                   ])
        s.points = s.points - [0, y_soma, 0]
        y_list += s.points[:, 1].tolist()

    y_max = np.max(y_list)
    y_min = np.min(y_list)

    if realign == True:

        if np.abs(y_min - y_max) > cortical_layer_thickness[animal][region]['L6']:
            print('Reducing size:')
            ratio = cortical_layer_thickness[animal][region]['L6'] / np.abs(y_min - y_max)
            print('ratio: ' + str(ratio))
            for s in m.iter():
                s.points = s.points * ratio
            print('segments reduced ')
            y_max = np.max(np.asarray(y_list) * ratio)
            y_min = np.min(np.asarray(y_list) * ratio)

        if y_max > 0:
            for s in m.iter():
                s.points = np.asarray([np.asarray([pts[0] / cortical_layer_thickness[animal][region]['L6'],
                                                   y_layer_percent(pts[1] - y_max, animal, region),
                                                   pts[2] / cortical_layer_thickness[animal][region]['L6']])
                                       for pts in s.points
                                       ])


        elif y_min < -cortical_layer_thickness[animal][region]['L6']:
            delta = y_min + cortical_layer_thickness[animal][region]['L6']
            for s in m.iter():
                s.points = np.asarray([np.asarray([pts[0] / cortical_layer_thickness[animal][region]['L6'],
                                                   y_layer_percent(pts[1] - delta, animal, region),
                                                   pts[2] / cortical_layer_thickness[animal][region]['L6']])
                                       for pts in s.points
                                       ])

        else:
            for s in m.iter():
                s.points = np.asarray([np.asarray([pts[0] / cortical_layer_thickness[animal][region]['L6'],
                                                   y_layer_percent(pts[1], animal, region),
                                                   pts[2] / cortical_layer_thickness[animal][region]['L6']])
                                       for pts in s.points
                                       ])
    else:
        for s in m.iter():
            s.points = np.asarray([np.asarray([pts[0] / cortical_layer_thickness[animal][region]['L6'],
                                               y_layer_percent(pts[1], animal, region),
                                               pts[2] / cortical_layer_thickness[animal][region]['L6']])
                                   for pts in s.points
                                   ])

    return m


def extract_moments(m, N=5):
    
    # if data_source == 'AIBS':
    seg_ax, seg_dend = arbor_segments(m)
    # if data_source == 'BBP':
    #     seg_ax, seg_dend = arbor_segments_BBP(m)
    
    dict_ = {'moments axon': [], 'moments dendrites': []}
#    print(seg_ax)
    for lim_sup, lim_inf in zip([5.0,-1.0, -2.0, -3.0, -4.0],
                                [-1.0, -2.0, -3.0, -4.0, -8.0]):
        
        msk_tmp_ax = [lim_sup > seg[1] > lim_inf for seg in seg_ax]
        mom_ax = compute_moments(seg_ax[msk_tmp_ax], N)
        msk_tmp_dend = [lim_sup > seg[1] > lim_inf for seg in seg_dend]
        mom_dend = compute_moments(seg_dend[msk_tmp_dend], N)
        dict_['moments axon'].append(mom_ax)
        dict_['moments dendrites'].append(mom_dend)
    
    return dict_

def gaussian_density(seg_ax, seg_dend):

    from scipy.stats import multivariate_normal

    k_ax = []
    k_dend = []

    Amp_dend = []
    Amp_ax = []

    for lim_sup, lim_inf in zip([5.0,-1.0, -2.0, -3.0, -4.0],
                                        [-1.0, -2.0, -3.0, -4.0, -8.0]):

        msk_tmp_ax = [lim_sup > seg[1] > lim_inf for seg in seg_ax]
        mom_ax_ = np.nan_to_num(compute_moments(seg_ax[msk_tmp_ax], N=10))
        msk_tmp_dend = [lim_sup > seg[1] > lim_inf for seg in seg_dend]
        mom_dend_ = np.nan_to_num(compute_moments(seg_dend[msk_tmp_dend], N=10))

        X_dend = np.nan_to_num(np.median(seg_dend[:,0][msk_tmp_dend]))
        Xstd_dend = np.nan_to_num(np.std(seg_dend[:,0][msk_tmp_dend]))
        Y_dend = np.nan_to_num(np.median(seg_dend[:,1][msk_tmp_dend]))
        Ystd_dend = np.nan_to_num(np.std(seg_dend[:,1][msk_tmp_dend]))
        Amp_dend.append(np.nan_to_num(np.sum(seg_dend[:,2][msk_tmp_dend])/np.sum(seg_dend[:,2])))

        X_ax = np.nan_to_num(np.median(seg_ax[:,0][msk_tmp_ax]))
        Xstd_ax = np.nan_to_num(np.std(seg_ax[:,0][msk_tmp_ax]))
        Y_ax = np.nan_to_num(np.median(seg_ax[:,1][msk_tmp_ax]))
        Ystd_ax = np.nan_to_num(np.std(seg_ax[:,1][msk_tmp_ax]))
        Amp_ax.append(np.nan_to_num(np.sum(seg_ax[:,2][msk_tmp_ax])/np.sum(seg_ax[:,2])))

        m_ax = (X_ax,Y_ax)
        m_dend = (X_dend,Y_dend)

        s_ax = np.asarray([[Xstd_ax, 0.],
                           [0., Ystd_ax]])
        s_dend = np.asarray([[Xstd_dend, 0.],
                             [0., Ystd_dend]])

        k_ax.append(multivariate_normal(mean=m_ax, cov=s_ax, allow_singular=True))
        k_dend.append(multivariate_normal(mean=m_dend, cov=s_dend, allow_singular=True))

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-1, 1)
    ylim = (-5, 0)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz_ax = np.sum([k1.pdf(xxyy) * A for k1,A in zip(k_ax, Amp_ax)],axis=0)
    zz_dend = np.sum([k1.pdf(xxyy) * A for k1,A in zip(k_dend, Amp_dend)],axis=0)

    return zz_ax, zz_dend

def moments_to_csv(ax_mom_data, dend_mom_data, ID_list, N=3):
    data_moments = []

    for mom_ax, mom_dend in zip(ax_mom_data, dend_mom_data):
        data_moments.append(mom_ax[0][0:N, 1].flatten().tolist() + mom_ax[1][0:N, 1].flatten().tolist() + mom_ax[2][0:N,
                                                                                                          1].flatten().tolist() +
                            mom_ax[3][0:N, 1].flatten().tolist() + mom_ax[4][0:N, 1].flatten().tolist() +
                            mom_ax[0][1:N, 0].flatten().tolist() + mom_ax[1][1:N, 0].flatten().tolist() + mom_ax[2][1:N,
                                                                                                          0].flatten().tolist() +
                            mom_ax[3][1:N, 0].flatten().tolist() + mom_ax[4][1:N, 0].flatten().tolist() +
                            mom_dend[0][0:N, 1].flatten().tolist() + mom_dend[1][0:N, 1].flatten().tolist() + mom_dend[
                                                                                                                  2][
                                                                                                              0:N,
                                                                                                              1].flatten().tolist() +
                            mom_dend[3][0:N, 1].flatten().tolist() + mom_dend[4][0:N, 1].flatten().tolist() +
                            mom_dend[0][1:N, 0].flatten().tolist() + mom_dend[1][1:N, 0].flatten().tolist() + mom_dend[
                                                                                                                  2][
                                                                                                              1:N,
                                                                                                              0].flatten().tolist() +
                            mom_dend[3][1:N, 0].flatten().tolist() + mom_dend[4][1:N, 0].flatten().tolist()
                            )
    data_moments_df = pd.DataFrame(np.asarray(data_moments),
                                   index=ID_list, columns=['morpho_moment_' + str(i)
                                                           for i in range(len(data_moments[0]))]).fillna(0)
    return data_moments_df


def neurom_extractor(path_to_morpho_collection, morpho_features):
    name_id = []
    feat_coll = []
    for name in os.listdir(path_to_morpho_collection):

        try:
            nrn = nm.load_morphology(path_to_morpho_collection + name)
            nrn_feats = []
            for feat in morpho_features:
                try:
                    nrn_feats.append(np.mean(nm.get(feat, nrn)))
                except:
                    nrn_feats.append(np.nan)

            feat_coll.append(np.asarray(nrn_feats))

        except:
            feat_coll.append(np.asarray([np.nan] * len(morpho_features)))
            print('No NM features for', name)

        name_id.append(name)

    neurom_features = pd.DataFrame(np.asarray(feat_coll),
                                   index=name_id,
                                   columns=['morpho_nm_' + feat_name for feat_name in morpho_features])
    return neurom_features
