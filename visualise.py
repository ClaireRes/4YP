#! usr/bin/env/ python

from mpl_toolkits.mplot3d import axes3d
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.spatial
from matplotlib import gridspec
import argparse
from scipy import signal

warnings.simplefilter(action='ignore', category=FutureWarning)


# Allows script to be run from command line & parses input files from command line
# or defaults to inputs given in script if none provided
def parse_inputs(fin, fin_gs, seq_len):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--gridsize')
    parser.add_argument('--length', type=int,
                        help="int length of recording to analyse (from start): 0 analyses whole recording")
    args = parser.parse_args()

    if args.file:
        fin = args.file
    if args.gridsize:
        fin_gs = args.gridsize
    if args.length:
        seq_len = args.length

    print('Data file in: %s' % fin)
    print('Grid size file in: %s' % fin_gs)
    print('Analysing sequence length: %d' % seq_len)

    return fin, fin_gs, seq_len


def get_grid_size(f_gs):
    gs = pd.read_csv(f_gs, header=None)
    row_n = gs.iloc[0, 0]
    col_n = gs.iloc[0, 1]
    dic = {'row_n': row_n, 'col_n': col_n}
    print('raw GRID SIZE = %d x %d' % (row_n, col_n))

    return dic


def zero_mean_data(stacked_arr):
    idx = int(stacked_arr[0, :, :].shape[1]/3)
    mean_x = np.mean(np.mean(stacked_arr[:, :, :idx], axis=0))
    mean_y = np.mean(np.mean(stacked_arr[:, :, idx:2*idx], axis=0))
    mean_z = np.mean(np.mean(stacked_arr[:, :, 2*idx:], axis=0))

    zerod_x = stacked_arr[:, :, :idx] - mean_x
    zerod_y = stacked_arr[:, :, idx:2*idx] - mean_y
    zerod_z = stacked_arr[:, :, 2*idx:] - mean_z

    zerod_stacked_arr = np.concatenate((zerod_x, zerod_y, zerod_z), axis=2)
    return zerod_stacked_arr


def get_avg_plane_normal(arr_3d):
    normals = []
    for n in range(arr_3d.shape[0]):
        tmp_split = np.split(arr_3d[n, :, :], 3, axis=1)
        x = tmp_split[0].flatten()
        y = tmp_split[1].flatten()
        z = tmp_split[2].flatten()
        flattened = np.array([x, y, z])
        u, s, vt = np.linalg.svd(flattened)
        normal = u[:, -1]
        normal = np.reshape(normal, (3, 1))
        normals.append(normal)

    stacked_normals = np.concatenate(normals, axis=1)
    avg_p_n = np.mean(stacked_normals, axis=1)

    return avg_p_n


def compute_rotation_matrix(a, b, rotation=1):
    R = np.array([[np.dot(a, b), -1*np.linalg.norm(np.cross(a, b)), 0],
                 [np.linalg.norm(np.cross(a, b)), np.dot(a, b), 0],
                 [0, 0, 1]])
    if rotation == 0:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = np.asmatrix(R)

    u = a
    v = b - (np.dot(a, b)*a)
    v = v/np.linalg.norm(v)
    w = np.cross(b, a)
    w = w/np.linalg.norm(w)

    Fi = np.array([u, v, w])
    Fi = np.asmatrix(Fi)
    Fi = Fi.T
    F = Fi.I

    U = np.matmul(Fi, np.matmul(R, F))
    return U


def rotate_data(data_arr, r_m):
    frames = []
    for n in range(data_arr.shape[0]):
        tmp_split = np.split(data_arr[n, :, :], 3, axis=1)
        tmp_stacked = np.stack(tmp_split, axis=0)

        x_vector = tmp_stacked[0, :, :].flatten()
        y_vector = tmp_stacked[1, :, :].flatten()
        z_vector = tmp_stacked[2, :, :].flatten()

        three_by_n = np.array([x_vector, y_vector, z_vector])
        rotated = np.matmul(r_m, three_by_n)
        rotated = np.array(rotated)

        x_grid = np.reshape(rotated[0, :], tmp_stacked[0, :, :].shape)
        y_grid = np.reshape(rotated[1, :], tmp_stacked[1, :, :].shape)
        z_grid = np.reshape(rotated[2, :], tmp_stacked[2, :, :].shape)

        frame = np.concatenate((x_grid, y_grid, z_grid), axis=1)
        frames.append(frame)

    frames_arr = np.stack(frames, axis=0)

    return frames_arr


def align_surface(df, num_frames):
    copy_df = df.copy()
    # drop external borders of missing values & fill in missing internal grid values
    drop_empty = copy_df.dropna(axis=0, how='all')
    dropped = drop_empty.dropna(axis=1, how='all')
    dropped = dropped.fillna(method='ffill')
    # reshape data into 3D array of slices of XYZ coordinates for each frame
    split_arr = np.split(dropped.values, num_frames, axis=1)
    stacked_arr = np.stack(split_arr, axis=0)
    stacked_arr2 = zero_mean_data(stacked_arr)

    p_n = get_avg_plane_normal(stacked_arr2)
    unit_p_n = p_n / np.linalg.norm(p_n)
    unit_p_n = unit_p_n.reshape((3,))
    # -1*k unit vector used so rotated chest surface points in positive z-direction
    k = np.array([0, 0, -1])

    r_m = compute_rotation_matrix(unit_p_n, k, rotation=1)
    frames = rotate_data(stacked_arr2, r_m)

    # To compare rotated vs unrotated plane set conpare=1
    compare = 0
    if compare == 1:
        r_m2 = compute_rotation_matrix(unit_p_n, k, rotation=0)
        frames2 = rotate_data(stacked_arr2, r_m2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        idx = int(frames[0, :, :].shape[1] / 3)
        # rotated

        ax.plot_surface(frames[0, :, :idx], frames[0, :, idx:2*idx],
                        frames[0, :, 2*idx:], cmap='Blues', alpha=0.7)
        # original surface orientation
        ax.plot_surface(frames2[0, :, :idx], frames2[0, :, idx:2 * idx],
                        frames2[0, :, 2 * idx:], cmap='Reds', alpha=0.7)

        ax.quiver(p_n[0], p_n[1], p_n[2], p_n[0]*50, p_n[1]*50, p_n[2]*50, color='red')
        ax.quiver(k[0], k[1], k[2], k[0] * 50, k[1] * 50, k[2] * 50, color='blue')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.show()
        sys.exit(0)
    else:
        return frames


# grid segmentation & volume calculation for right-left chest-abdomen segments
def partition_data(frames_arr, delay):
    grid_s = frames_arr[0, :, :].shape
    grid_s = (grid_s[0], int(grid_s[1]/3))

    height_axis = int(np.argmax(grid_s))
    height = grid_s[height_axis]
    breadth = grid_s[int(np.argmin(grid_s))]
    print('True grid size: [%d x %d]' % (height, breadth))

    xmax = np.amax(frames_arr[:, :, :grid_s[1]])
    xmin = np.amin(frames_arr[:, :, :grid_s[1]])
    ymax = np.amax(frames_arr[:, :, grid_s[1]:2*grid_s[1]])
    ymin = np.amin(frames_arr[:, :, grid_s[1]:2*grid_s[1]])

    zmax = np.amax(frames_arr[:, :, 2*grid_s[1]:])
    zmin = np.amin(frames_arr[:, :, 2*grid_s[1]:])

    # split into x, y, z arrays
    data = np.split(frames_arr, 3, axis=2)
    data_ind = ['x', 'y', 'z']
    counter = 0
    split_data = {}
    for ax in data:
        # split into left-right
        if height_axis == 0:
            split = (ax[:, :, :int(grid_s[1]/2)+1], ax[:, :, int(grid_s[1]/2):])
            h1 = split[0]
            h2 = split[1]
            qs1 = (h1[:, :int(grid_s[0]/2)+1, :], h1[:, int(grid_s[0]/2):, :])
            q1 = qs1[0]
            q3 = qs1[1]
            qs2 = (h2[:, :int(grid_s[0]/2)+1, :], h2[:, int(grid_s[0]/2):, :])
            q2 = qs2[0]
            q4 = qs2[1]
        else:
            split = (ax[:, :int(grid_s[0]/2)+1, :], ax[:, int(grid_s[0]/2):, :])
            h1 = split[0]
            h2 = split[1]
            qs1 = (h1[:, :, :int(grid_s[1]/2)+1], h1[:, :, int(grid_s[1]/2):])
            q1 = qs1[0]
            q3 = qs1[1]
            qs2 = (h2[:, :, :int(grid_s[1] / 2)+1], h2[:, :, int(grid_s[1]/2):])
            q2 = qs2[0]
            q4 = qs2[1]

        tmp_list = [q1, q2, q3, q4]
        split_data[data_ind[counter]] = tmp_list
        counter += 1

    # estimate grid orientation from finding chest (peaks in z)
    q1_mean = np.mean(np.mean(split_data['z'][0], axis=0))
    q2_mean = np.mean(np.mean(split_data['z'][1], axis=0))
    q3_mean = np.mean(np.mean(split_data['z'][2], axis=0))
    q4_mean = np.mean(np.mean(split_data['z'][3], axis=0))

    if (q1_mean+q2_mean) > (q3_mean+q4_mean):
        labels = {'q1': 'R Chest', 'q2': 'L Chest', 'q3': 'R Abdomen',
                  'q4': 'L Abdomen'}
    else:
        labels = {'q4': 'R Chest', 'q3': 'L Chest', 'q2': 'R Abdomen',
                  'q1': 'L Abdomen'}
    # volume calculations
    v_l1 = []
    v_l2 = []
    v_l3 = []
    v_l4 = []
    t_list = []
    prev_v1 = prev_v2 = prev_v3 = prev_v4 = 0
    for n in range(split_data['x'][0].shape[0]):
        v1 = calculate_volume(split_data['x'][0][n, :, :],
                              split_data['y'][0][n, :, :],
                              split_data['z'][0][n, :, :], prev_v1)
        prev_v1 = v1
        v2 = calculate_volume(split_data['x'][1][n, :, :],
                              split_data['y'][1][n, :, :],
                              split_data['z'][1][n, :, :], prev_v2)
        prev_v2 = v2
        v3 = calculate_volume(split_data['x'][2][n, :, :],
                              split_data['y'][2][n, :, :],
                              split_data['z'][2][n, :, :], prev_v3)
        prev_v3 = v3
        v4 = calculate_volume(split_data['x'][3][n, :, :],
                              split_data['y'][3][n, :, :],
                              split_data['z'][3][n, :, :], prev_v4)
        prev_v4 = v4
        v_l1.append(v1)
        v_l2.append(v2)
        v_l3.append(v3)
        v_l4.append(v4)
        t_list.append(n*delay)

    v_l1 = zero_offset_vol(v_l1)
    v_l2 = zero_offset_vol(v_l2)
    v_l3 = zero_offset_vol(v_l3)
    v_l4 = zero_offset_vol(v_l4)

    tup = (split_data, labels, v_l1, v_l2, v_l3, v_l4, t_list, zmax,
           zmin, xmax, xmin, ymax, ymin)

    return tup


# Remove offset in volume sequences before scaling
def zero_offset_vol(v_list):
    vl = np.asarray(v_list)
    vl_zerod = vl - np.amin(vl)
    v_list_zerod = vl_zerod.tolist()

    return v_list_zerod


# approximate volume enclosed by surface & z=0 plane using Delaunay triangulation
# of surface and volume of prisms formed by these triangles
def calculate_volume(x_arr, y_arr, z_arr, prev_v):
    x = x_arr.flatten()
    y = y_arr.flatten()
    z = z_arr.flatten()

    xyz = np.vstack((x, y, z)).T
    try:
        d = scipy.spatial.Delaunay(xyz[:, :2])
        tri = xyz[d.simplices]
        a = tri[:, 0, :2] - tri[:, 1, :2]
        b = tri[:, 0, :2] - tri[:, 2, :2]
        proj_area = np.cross(a, b).sum(axis=-1)
        zavg = tri[:, :, 2].sum(axis=1)
        vol = zavg * np.abs(proj_area) / 6.0
        return vol.sum()
    except scipy.spatial.qhull.QhullError:
        print('Invalid volume due to Qhull error - using previous v value')
        return prev_v


# Use Savitzky Golday filter with polynomial order poly_d and segment length seg_l
# to smooth estimated volume sequences
def smooth_volume(v_list):
    poly_d = 3
    seg_l = 25
    smoothed = signal.savgol_filter(v_list, seg_l, poly_d)
    return smoothed


def scale_volume(v1, v2, v3, v4):
    total_v = np.add(np.add(v1, v2), np.add(v3, v4))
    max_v = np.max(total_v)
    # scale values so total volume lies between 0-1
    v1_scaled = v1 / max_v
    v2_scaled = v2 / max_v
    v3_scaled = v3 / max_v
    v4_scaled = v4 / max_v

    return v1_scaled, v2_scaled, v3_scaled, v4_scaled


# Reads surface data, calls interpolation, rotation & volume calculation routines
# Returns tuple of values required for plotting
def get_frame_data(f_data, gs, delay, frame_subset=0):
    start_time = time.time()
    # to read part of dataset
    if frame_subset > 0:
        c_idx = gs['col_n']*3*frame_subset
        try:
            df = pd.read_csv(f_data, header=None, usecols=np.linspace(0,
                                                                      c_idx-1,
                                                                      c_idx,
                                                                      dtype=int))
        except ValueError:
            print('Sample too short for given frame subset')
            print('Reading entire sample instead')
            df = pd.read_csv(f_data, header=None)
    # to read in all the data
    else:
        df = pd.read_csv(f_data, header=None)

    num_frames = int(len(df.columns)/(gs['col_n']*3))
    print('NUMBER OF FRAMES: %d' % num_frames)

    # normalisation of surface position & orientation
    frames = align_surface(df, num_frames)

    # segmentation of surface
    data_dic,lbls,v1,v2,v3,v4,tl,zmx,zmn,xmx,xmn,ymx,ymn = partition_data(frames,
                                                                          delay)
    print('Time to read in data: {} s'.format(time.time() - start_time))

    return data_dic,lbls,v1,v2,v3,v4,tl,zmx,zmn,xmx,xmn,ymx,ymn


# Calculate average respiratory rate in breaths/min for recording
def calculate_rr(vlist, tlist):
    peaks, _ = signal.find_peaks(vlist)
    num_peaks = peaks.shape[0]
    tt_mins = tlist[-1]/60
    rr = num_peaks/tt_mins
    print('Average respiratory rate = %.1f breaths per minute' % rr)
    return rr


# Dynamic plotting of segmented surface over time and extracted relative volume sequences
def comparison_plot(t_list, vlist1, vlist2, vlist3, vlist4, data, lbls,
                    delay, zmax, zmin, xmx, xmn, ymx, ymn):
    x_q1 = data['x'][0]
    x_q2 = data['x'][1]
    x_q3 = data['x'][2]
    x_q4 = data['x'][3]
    y_q1 = data['y'][0]
    y_q2 = data['y'][1]
    y_q3 = data['y'][2]
    y_q4 = data['y'][3]
    z_q1 = data['z'][0]
    z_q2 = data['z'][1]
    z_q3 = data['z'][2]
    z_q4 = data['z'][3]

    fig = plt.figure(figsize=(9, 14))
    gs = gridspec.GridSpec(4, 4)
    ax1 = fig.add_subplot(gs[:-1, :-1], projection='3d')
    ax2 = fig.add_subplot(gs[3, :])

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Volume (Arbitrary units)')

    # to set axis limits
    ax1.set_zlim(zmin-20, zmax+20)
    ax1.set_xlim(xmx, xmn)
    ax1.set_ylim(ymx, ymn)

    surface1 = None
    surface2 = None
    surface3 = None
    surface4 = None
    cursor1 = None
    smoothed1 = smooth_volume(vlist1)
    smoothed2 = smooth_volume(vlist2)
    smoothed3 = smooth_volume(vlist3)
    smoothed4 = smooth_volume(vlist4)

    ss1, ss2, ss3, ss4 = scale_volume(smoothed1, smoothed2, smoothed3, smoothed4)

    if lbls['q1'] == 'R Chest':
        r_total = ss1 + ss3
        l_total = ss2 + ss4
    else:
        l_total = ss1 + ss3
        r_total = ss2 + ss4

    for n in range(x_q1.shape[0]):
        if surface1:
            ax1.collections.remove(surface1)
            ax1.collections.remove(surface2)
            ax1.collections.remove(surface3)
            ax1.collections.remove(surface4)
        else:
            ax2.plot(t_list, ss1, color='blue', alpha=0.5, label=lbls['q1'])
            ax2.plot(t_list, ss2, color='green', alpha=0.5,
                     label=lbls['q2'])
            ax2.plot(t_list, ss3, color='red', alpha=0.5, label=lbls['q3'])
            ax2.plot(t_list, ss4, color='yellow', alpha=0.5,
                     label=lbls['q4'])
            ax2.plot(t_list, r_total, color='purple', alpha=0.5,
                     label='Right total')
            ax2.plot(t_list, l_total, color='black', alpha=0.5,
                     label='Left total')

            # to plot unsmoothed v sequences
            """
            ax2.plot(t_list, vlist1, color='blue', alpha=0.1)
            ax2.plot(t_list, vlist2, color='green', alpha=0.1)
            ax2.plot(t_list, vlist3, color='red', alpha=0.1)
            ax2.plot(t_list, vlist4, color='yellow', alpha=0.1)
            """
            ax2.legend(loc='right')

        if cursor1:
            cursor1.pop(0).remove()

        surface1 = ax1.plot_surface(x_q1[n, :, :],
                                    y_q1[n, :, :],
                                    z_q1[n, :, :], cmap='Blues')
        surface2 = ax1.plot_surface(x_q2[n, :, :],
                                    y_q2[n, :, :],
                                    z_q2[n, :, :], cmap='Greens')
        surface3 = ax1.plot_surface(x_q3[n, :, :],
                                    y_q3[n, :, :],
                                    z_q3[n, :, :], cmap='Reds')
        surface4 = ax1.plot_surface(x_q4[n, :, :],
                                    y_q4[n, :, :],
                                    z_q4[n, :, :], cmap='YlOrBr')

        cursor1 = ax2.plot(t_list[n], r_total[n], marker='x', ms=10, c='red')
        plt.pause(delay)

    plt.show()


def main(delay, f_data, f_gs, seq_length):
    f_data, f_gs, seq_length = parse_inputs(f_data, f_gs, seq_length)

    gs = get_grid_size(f_gs)

    print('Collecting data')
    data_dic,lbls,v1,v2,v3,v4,tl,zmx,zmn,xmx,xmn,ymx,ymn = get_frame_data(f_data,
                                                                          gs,
                                                                          delay,
                                                                          frame_subset=seq_length)

    print('\nFinished collecting data')

    input('Press Enter to continue...')

    print('Plotting data')
    plot_ts = time.time()
    comparison_plot(tl, v1, v2, v3, v4, data_dic, lbls, delay, zmx, zmn,xmx,xmn,ymx,ymn)
    print('Visualisation running time = {}'.format(time.time() - plot_ts))


if __name__ == "__main__":
    try:
        print('Working directory: ', os.getcwd())
        f_in = './slp_demo_file.csv'
        f_in_gs = './slp_demo_file_gs.csv'
        td = 1/25
        sequence_length = 2000
        main(td, f_in, f_in_gs, sequence_length)
    except FileNotFoundError as e:
        print('Problem with file path')
        print(str(e))
