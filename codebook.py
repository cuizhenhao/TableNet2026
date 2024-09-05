import math

import kmeans1d
import numpy as np
import torch
from sklearn.metrics import r2_score


# TODO linear 层还有个 feature_map_count = int(self.in_features / self.previous_codebook.shape[0])，需要考虑是否有影响
def decode(input: torch.Tensor, codebook):  #用来解码（从codebook中读出输入的真实值）
    original_input = input.cpu().numpy().astype(np.uint8)
    decoded_input = np.zeros_like(input, dtype=np.float32)
    feature_map_count = max(1, int(input.shape[1] / codebook.shape[0]))
    for in_ch in range(input.shape[1]):
        decoded_input[:, in_ch] = codebook[int(in_ch/feature_map_count)][1][original_input[:, in_ch]]
    return torch.Tensor(decoded_input)


# def encode(input: torch.Tensor, codebook):  #用来编码（将输入编码为codebook值）
#     encode_input = input.clone()
#     for in_ch in range(codebook.shape[0]):
#         origin_shape = input[:, in_ch].shape
#         on_ch = encode_input[:, in_ch].reshape(-1)
#         # on_ch = nearest_value_batch(on_ch,codebook[in_ch][1])
#         for i in range(on_ch.shape[0]):
#             on_ch[i] = nearest_value(on_ch[i].cpu().numpy().astype(np.float32),codebook[in_ch][1])
#         encode_input[:, in_ch] = torch.from_numpy(on_ch).reshape(origin_shape)
#     encode_input = encode_input.cpu().numpy().astype(np.uint8)
#     return torch.Tensor(encode_input)
#
# # 得到value 与 最接近的聚类中心（index）
# def nearest_value(value: int, cluster: np.array):
#     return (np.abs(cluster - value)).argmin()


def encode(input: torch.Tensor, codebook: np.ndarray):

    encode_input = input.clone()
    for in_ch in range(codebook.shape[0]):
        origin_shape = input[:, in_ch].shape
        on_ch = encode_input[:, in_ch].reshape(-1)
        codebook_tensor = torch.from_numpy(codebook[in_ch][1])
        abs_diff = torch.abs(on_ch[:, None] - codebook_tensor)

        # Find the index of the nearest value using 'argmin'
        nearest_indices = torch.argmin(abs_diff, dim=1)

        # Update 'on_ch' using 'nearest_indices'
        on_ch[:] = nearest_indices
        # on_ch[:] = codebook_tensor[nearest_indices]
        # for i in range(on_ch.shape[0]):
        #     on_ch[i] = nearest_value(on_ch[i], torch.from_numpy(codebook[in_ch][1]))

        encode_input[:, in_ch] = on_ch.reshape(origin_shape)

    encode_input = encode_input.to(torch.int32)
    return encode_input


def nearest_value(value, cluster):
    return torch.argmin(torch.abs(cluster - value))

def nearest_value_batch(values, cluster):
    # make sure array is a numpy array
    values = np.atleast_1d(values)
    # indices = np.zeros(values.shape[0])
    # indices_map = {}
    # for i in range(values.shape[0]):
    #     value =  values[i]
    #     if value not in indices_map:
    #         indices_map[value] = nearest_value(value, cluster)
    #     indices[i] = indices_map[value]
    indices = np.abs(np.subtract.outer(cluster, values)).argmin(0)
    return indices


# 得到value 与 最接近的聚类中心的值（value）
# `array` should be sorted
def get_closest(values, array):
    # make sure array is a numpy array
    array = np.array(array)
    values = np.array(values)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array)) | (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(
        values - array[np.minimum(idxs, len(array) - 1)])))
    idxs[prev_idx_is_less] -= 1

    return array[idxs]

def dict_reverse(l, size):
    result = [[] for i in range(size)]
    for i in range(len(l)):
        value = l[i]
        result[value].append(i)
    return result

def cal_output_merge_dict(l1, l2, size):
    dict1 = dict_reverse(l1,size)
    dict2 = dict_reverse(l2,size)
    l3 = [-1 for i in range(len(l1))]
    dict3 = []
    for i in range(len((l1))):
        if l3[i] == -1:
            merge_one = [i]
            done = []
            x,max=0,len(merge_one)
            while x < max:
                j = merge_one[x]
                if j in done:
                    x+=1
                    continue
                done.append(j)
                merge_one.extend(dict1[l1[j]])
                merge_one.extend(dict2[l2[j]])
                merge_one = list(set(merge_one))
                merge_one.sort()
                if len(merge_one)>max:
                    x=0
                else:
                    x+=1
                max = len(merge_one)
            dict3.append(list(set(merge_one)))
            for j in merge_one:
                l3[j] = len(dict3)
    all_len = 0
    for l in dict3:
        all_len += len(l)
    assert all_len == len(l1)
    return l3,dict3


delete_size = 0
all_size = 0
def cluster_weight(name, weight):
    global delete_size
    global all_size
    origin_shape, origin_weight = weight.shape, weight.clone()
    weight = weight.view(weight.size(0), -1)
    clusters_list = []
    dict_list = []
    for i in range(weight.shape[1]):
        x11 = weight[:, i].cpu().detach().numpy().reshape(-1)
        all_len = (x11!=0).sum()
        all_size = all_size+all_len
        if all_len < 16:
            continue
        clusters, centroids = kmeans1d.cluster(x11, all_len //2)
        delete_size+=all_len-len(centroids)
        centroids = np.array(centroids)
        clusters = np.array(clusters)
        clusters_list.append(clusters)
        # if len(clusters_list)>=2:
        #     dicts = cal_output_merge_dict(clusters_list[-2],clusters_list[-1],len(centroids))
        #     dict_list.append(dicts)
        after_cluster_x11 = centroids[clusters]
        weight[:, i] = torch.from_numpy(after_cluster_x11)
    weight = weight.reshape(origin_shape)
    r2s = r2_score(weight.cpu().detach().numpy().reshape(-1), origin_weight.cpu().detach().numpy().reshape(-1))
    print(name, r2s, delete_size,all_size)
    return clusters_list,dict_list

def cluster_codebook(codebook, size):
    origin_shape, origin_codebook = codebook.shape, codebook.clone()
    codebook = codebook.view(-1)
    # x11 = codebook.cpu().detach().numpy().reshape(-1)
    clusters, centroids = kmeans1d.cluster(codebook, size)
    centroids = np.array(centroids)
    clusters = np.array(clusters)
    after_cluster_x11 = centroids[clusters]
    codebook = torch.from_numpy(after_cluster_x11)
    codebook = codebook.reshape(origin_shape)
    # r2s = r2_score(codebook.cpu().detach().numpy().reshape(-1), origin_codebook.cpu().detach().numpy().reshape(-1))
    # print(r2s)
    return centroids


def order_weight(weight):
    abs_weight = torch.abs(weight)
    _, sort_index = torch.sort(abs_weight, dim=1)
    return sort_index
