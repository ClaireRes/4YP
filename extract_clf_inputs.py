#! usr/bin/env/ python

import warnings
import os
import glob
import csv
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.spatial
from scipy import signal

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_grid_size(f_gs):
    grid_s = pd.read_csv(f_gs, header=None)
    row_n = grid_s.iloc[0, 0]
    col_n = grid_s.iloc[0, 1]
    dic = {'row_n': row_n, 'col_n': col_n}

    return dic


def get_frame_data(frame_subset, gs, f_data):
    if frame_subset > 0:
        c_idx = gs['col_n'] * 3 * frame_subset
        try:
            df = pd.read_csv(f_data, header=None, usecols=np.linspace(0, c_idx - 1,
                                                                    c_idx, dtype=int))
        except ValueError:
            print('Sample too short for given frame subset')
            df = pd.read_csv(f_data, header=None)
    # to read in all whole recording
    else:
        df = pd.read_csv(f_data, header=None)

    return df


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

    return frames


def calculate_volume(x_arr, y_arr, z_arr, prev_v):
    x = x_arr.flatten()
    y = y_arr.flatten()
    z = z_arr.flatten()

    xyz = np.vstack((x, y, z)).T
    try:
        d = scipy.spatial.Delaunay(xyz[:, :2])
        tri = xyz[d.vertices]

        a = tri[:, 0, :2] - tri[:, 1, :2]
        b = tri[:, 0, :2] - tri[:, 2, :2]
        proj_area = np.cross(a, b).sum(axis=-1)
        zavg = tri[:, :, 2].sum(axis=1)
        vol = zavg * np.abs(proj_area) / 6.0

        return vol.sum()
    except scipy.spatial.qhull.QhullError:
        print('Qhull error')
        return prev_v


def zero_offset_vol(v_list):
    vl = np.asarray(v_list)
    vl_zerod = vl - np.amin(vl)
    v_list_zerod = vl_zerod.tolist()

    return v_list_zerod


def smooth_volume(v_list):
    smoothed = signal.savgol_filter(v_list, 25, 3)
    return smoothed


def scale_volume(v1, v2, v3, v4):
    total_v = np.add(np.add(v1, v2), np.add(v3, v4))
    max_v = np.max(total_v)

    # scale data so total volume lies between 0-1
    v1_scaled = v1/max_v
    v2_scaled = v2/max_v
    v3_scaled = v3/max_v
    v4_scaled = v4/max_v

    return v1_scaled, v2_scaled, v3_scaled, v4_scaled


def partition_data(frames_arr):
    grid_s = frames_arr[0, :, :].shape
    grid_s = (grid_s[0], int(grid_s[1]/3))

    height_axis = int(np.argmax(grid_s))
    height = grid_s[height_axis]
    breadth = grid_s[int(np.argmin(grid_s))]
    print('True grid size: [%d x %d]' % (height, breadth))

    # split into x, y, z arrays
    data = np.split(frames_arr, 3, axis=2)
    data_ind = ['x', 'y', 'z']
    counter = 0
    split_data = {}
    for ax in data:
        # split into left-right
        if height_axis == 0:
            split = (ax[:, :, :int(grid_s[1] / 2) + 1], ax[:, :, int(grid_s[1] / 2):])
            h1 = split[0]
            h2 = split[1]
            qs1 = (h1[:, :int(grid_s[0] / 2) + 1, :], h1[:, int(grid_s[0] / 2):, :])
            q1 = qs1[0]
            q3 = qs1[1]
            qs2 = (h2[:, :int(grid_s[0] / 2) + 1, :], h2[:, int(grid_s[0] / 2):, :])
            q2 = qs2[0]
            q4 = qs2[1]
        else:
            split = (ax[:, :int(grid_s[0] / 2) + 1, :], ax[:, int(grid_s[0] / 2):, :])
            h1 = split[0]
            h2 = split[1]
            qs1 = (h1[:, :, :int(grid_s[1] / 2) + 1], h1[:, :, int(grid_s[1] / 2):])
            q1 = qs1[0]
            q3 = qs1[1]
            qs2 = (h2[:, :, :int(grid_s[1] / 2) + 1], h2[:, :, int(grid_s[1] / 2):])
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
    prev_v1 = prev_v2 = prev_v3 = prev_v4 = 0
    for n in range(split_data['x'][0].shape[0]):
        v1 = calculate_volume(split_data['x'][0][n, :, :],
                              split_data['y'][0][n, :, :],
                              split_data['z'][0][n, :, :], prev_v1)
        v2 = calculate_volume(split_data['x'][1][n, :, :],
                              split_data['y'][1][n, :, :],
                              split_data['z'][1][n, :, :], prev_v2)
        v3 = calculate_volume(split_data['x'][2][n, :, :],
                              split_data['y'][2][n, :, :],
                              split_data['z'][2][n, :, :], prev_v3)
        v4 = calculate_volume(split_data['x'][3][n, :, :],
                              split_data['y'][3][n, :, :],
                              split_data['z'][3][n, :, :], prev_v4)
        prev_v4 = v4
        v_l1.append(v1)
        v_l2.append(v2)
        v_l3.append(v3)
        v_l4.append(v4)

    v_l1 = zero_offset_vol(v_l1)
    smoothed1 = smooth_volume(v_l1)
    v_l2 = zero_offset_vol(v_l2)
    smoothed2 = smooth_volume(v_l2)
    v_l3 = zero_offset_vol(v_l3)
    smoothed3 = smooth_volume(v_l3)
    v_l4 = zero_offset_vol(v_l4)
    smoothed4 = smooth_volume(v_l4)

    ss1, ss2, ss3, ss4 = scale_volume(smoothed1, smoothed2, smoothed3, smoothed4)

    # append to dic with ordering so q1 is always the right chest
    if labels['q1'] == 'R Chest':
        v_ts = {'q1': ss1, 'q2': ss2, 'q3': ss3, 'q4': ss4}
    else:
        v_ts = {'q1': ss4, 'q2': ss3, 'q3': ss2, 'q4': ss1}

    return v_ts


def extract_features(f_in, f_gs, slice_length, extracted_df=0, slice_num=0,
                     whole_df=pd.DataFrame()):
    grid_s = get_grid_size(f_gs)
    if extracted_df == 0:
        df = get_frame_data(slice_length, grid_s, f_in)
    else:
        copy = whole_df.copy()
        idx_start = slice_num*slice_length*3*grid_s['col_n']
        df = copy.iloc[:, idx_start:(idx_start+(slice_length*3*grid_s['col_n']))]

    num_frames = int(len(df.columns) / (grid_s['col_n'] * 3))
    frames = align_surface(df, num_frames)
    vols_dic = partition_data(frames)

    return vols_dic


def get_slices_in_file(f_in, f_gs, slice_length):
    grid_s = get_grid_size(f_gs)
    df = pd.read_csv(f_in, header=None)
    num_slices = df.shape[1] // (slice_length*3*grid_s['col_n'])

    return num_slices, df, grid_s


def main():
    # set 0 for normal, 1 for RD
    label = 0
    # split_samples=0 extracts first slice from sample, 1 extracts as many slices
    # of slice_length as possible from single sample
    split_samples = 1
    slice_length = 2000

    # adjust as necessary for path of pn3 files
    file_path = os.getcwd()
    # name of folder to store extracted segment-volume data in
    results_name = 'results'

    results_path = os.path.join(file_path, results_name + '/')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    print("Storing extracted results in directory ", results_path)

    os.chdir(file_path)

    # creates CSVs to store segment data, sample names and labels for classifier input
    f_out_1 = results_path + results_name + '_q1.csv'
    f_out_2 = results_path + results_name +'_q2.csv'
    f_out_3 = results_path + results_name +'_q3.csv'
    f_out_4 = results_path + results_name + '_q4.csv'
    f_out_names = results_path + results_name + '_names.csv'
    f_out_labels = results_path + results_name + '_labels.csv'
    pn3 = 'pn3'

    a_names = [i.rsplit('.', 2)[0] for i in glob.glob('*.{}'.format(pn3))]

    with open(f_out_1, 'w', newline='') as csvfile1, \
        open(f_out_2, 'w', newline='') as csvfile2,  \
        open(f_out_3, 'w', newline='') as csvfile3,  \
        open(f_out_4, 'w', newline='') as csvfile4,  \
        open(f_out_names, 'w', newline='') as csvfilen, \
        open(f_out_labels, 'w', newline='') as csvfilelb:

            writer1 = csv.writer(csvfile1)
            writer2 = csv.writer(csvfile2)
            writer3 = csv.writer(csvfile3)
            writer4 = csv.writer(csvfile4)
            writern = csv.writer(csvfilen)
            writerlb = csv.writer(csvfilelb)

            keys = ['q1', 'q2', 'q3', 'q4']

            for name in a_names:
                f_gs = './' + name + '_gs.csv'
                f = './'  + name + '.csv'
                print('Extracting %s' % f)
                if split_samples == 0:
                    features_dic = extract_features(f, f_gs, slice_length)
                    writer1.writerow(features_dic[keys[0]])
                    writer2.writerow(features_dic[keys[1]])
                    writer3.writerow(features_dic[keys[2]])
                    writer4.writerow(features_dic[keys[3]])
                    writern.writerow([f])
                    writerlb.writerow([label])
                else:
                    n_slices, df_full, grid_s = get_slices_in_file(f,
                                                                   f_gs,
                                                                   slice_length)
                    print('%d slices of length %d found' % (n_slices,
                                                            slice_length))
                    if n_slices < 1:
                        print('No slices long enough in sample')
                        print('Using slice length = %d' % df_full.shape[1])
                        s_length = df_full.shape[1] // 3*grid_s['col_n']
                        n_slices = 1
                    else:
                        s_length = slice_length
                    for s in range(n_slices):
                        print('Extracting slice %d' % s)
                        f_name = './' + name + '_slice' + str(s) + '.csv'
                        features_dic = extract_features(f, f_gs, s_length,
                                                        extracted_df=1,
                                                        slice_num=s,
                                                        whole_df=df_full)
                        writer1.writerow(features_dic[keys[0]])
                        writer2.writerow(features_dic[keys[1]])
                        writer3.writerow(features_dic[keys[2]])
                        writer4.writerow(features_dic[keys[3]])
                        writern.writerow([f_name])
                        writerlb.writerow([label])


if __name__ == "__main__":
    main()
