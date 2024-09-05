import numpy as np

def bindigits(n, bits):
    s = bin(n & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)


def high4bit(n):
    return (n & 0xf0) >> 4

def low4bit(n):
    return n & 0x0f

def l4_h4bit(n):
    # 低三位
    l = low4bit(n)
    # 高四位
    h = high4bit(n)
    return  l,h

def l3_h4bit(n):
    # 低三位
    low_three_bits = n & 7
    # 高四位
    high_four_bits = n >> 3 & 15
    return  low_three_bits, high_four_bits

def l4_h3bit(n):
    # 低四位
    low_4_bits = n & 15
    # 高三位
    high_3_bits = n >> 4 & 7
    return  low_4_bits, high_3_bits

# 平均化聚类
def mean_cluster(x, n_clusters):
    min, max = x.min(), x.max()
    step = (max - min) / n_clusters
    if max != min:
        cluster = np.arange(min, max, step, dtype=float)
    else:
        cluster = np.array([min])
    n_clusters = len(cluster)
    return (n_clusters, cluster, min, step)


if __name__ == '__main__':
    print(bindigits(200, 8))
    print(high4bit(200))
    print(low4bit(200))
    print(high4bit(np.array([1,200,255])))
    print(low4bit(np.array([1,200,255])))