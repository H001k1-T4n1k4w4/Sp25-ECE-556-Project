import cv2 as cv
import numpy as np
import heapq
import os
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple
from math import log2

def closest_power_of_2(n: int) -> int:
    r = 1
    while r * 2 <= n:
        r *= 2
    return r

def ind_initialize(max_size: int, N: int, step: int) -> List[int]:
    ind_set = []
    ind = N
    while ind < max_size - N:
        ind_set.append(ind)
        ind += step
    if ind_set and ind_set[-1] < max_size - N - 1:
        ind_set.append(max_size - N - 1)
    return ind_set

def precompute_BM(img: np.ndarray,
                  #width: int,
                  #height: int,
                  kHW: int,
                  NHW: int,
                  nHW: int,
                  pHW: int,
                  tauMatch: float) -> List[List[int]]:
    # Ensure img is flattened and correct size
    height, width = img.shape
    img2d = img
    img = img.astype(np.float32).flatten() / 255.0 

    assert len(img) == width * height, f"Image size mismatch: got {len(img)}, expected {width * height}"
    
    # Declarations
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW
    diff_table = np.zeros(width * height, dtype=np.float32)
    sum_table = np.full(((nHW + 1) * Ns, width * height), 2 * threshold, dtype=np.float32)
    patch_table = [[] for _ in range(width * height)]
    
    # Initialize indices
    row_ind = ind_initialize(height - kHW + 1, nHW, pHW)
    column_ind = ind_initialize(width - kHW + 1, nHW, pHW)

    # For each possible distance, precompute inter-patches distance
    for di in range(nHW + 1):
        for dj in range(Ns):
            dk = int(di * width + dj) - int(nHW)
            ddk = di * Ns + dj

            # Process the image containing the square distance between pixels
            for i in range(nHW, height - nHW):
                k = i * width + nHW
                for j in range(nHW, width - nHW):
                    diff_table[k] = (float(img[k + dk]) - float(img[k])) ** 2
                    k += 1

            # Compute the sum for each patches, using integral images
            dn = nHW * width + nHW
            
            # 1st patch, top left corner
            value = 0.0
            for p in range(kHW):
                pq = p * width + dn
                for q in range(kHW):
                    value += diff_table[pq]
                    pq += 1
            sum_table[ddk, dn] = value

            # 1st row, top
            for j in range(nHW + 1, width - nHW):
                ind = nHW * width + j - 1
                sum_val = sum_table[ddk, ind]
                for p in range(kHW):
                    sum_val += diff_table[ind + p * width + kHW] - diff_table[ind + p * width]
                sum_table[ddk, ind + 1] = sum_val

            # General case
            for i in range(nHW + 1, height - nHW):
                ind = (i - 1) * width + nHW
                sum_val = sum_table[ddk, ind]
                
                # 1st column, left
                for q in range(kHW):
                    sum_val += diff_table[ind + kHW * width + q] - diff_table[ind + q]
                sum_table[ddk, ind + width] = sum_val

                # Other columns
                k = i * width + nHW + 1
                pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1
                for j in range(nHW + 1, width - nHW):
                        
                    sum_table[ddk, k] = (sum_table[ddk, k - 1] +
                                        sum_table[ddk, k - width] -
                                        sum_table[ddk, k - 1 - width] +
                                        diff_table[pq] -
                                        diff_table[pq - kHW] -
                                        diff_table[pq - kHW * width] +
                                        diff_table[pq - kHW - kHW * width])
                    k += 1
                    pq += 1

    # Precompute Block Matching
    for ind_i in range(len(row_ind)):
        for ind_j in range(len(column_ind)):
            # Initialization
            k_r = row_ind[ind_i] * width + column_ind[ind_j]
            table_distance = []
            patch_table[k_r] = []
            positions_processed += 1

            # Threshold distances to keep similar patches
            for dj in range(-nHW, nHW + 1):
                # Positive di
                for di in range(nHW + 1):
                    if sum_table[dj + nHW + di * Ns, k_r] < threshold:
                        k_di = k_r + di * width + dj
                        table_distance.append(
                            (float(sum_table[dj + nHW + di * Ns, k_r]), k_di))

                # Negative di
                for di in range(-nHW, 0):
                    if sum_table[-dj + nHW + (-di) * Ns, k_r] < threshold:
                        k_di = k_r + di * width + dj
                        table_distance.append(
                            (float(sum_table[-dj + nHW + (-di) * Ns, k_di]), k_di))

            # Always include self-patch if no matches found
            if len(table_distance) == 0:
                print(f"No matches found for position {k_r} ({k_r//width}, {k_r%width})")
                table_distance.append((0.0, k_r))

            # We need a power of 2 for the number of similar patches,
            # because of the Welsh-Hadamard transform on the third dimension.
            # We assume that NHW is already a power of 2
            nSx_r = closest_power_of_2(len(table_distance)) if NHW > len(table_distance) else NHW

            # To avoid problem
            if nSx_r == 1 and len(table_distance) == 0:
                print("problem size")
                table_distance.append((0.0, k_r))

            # Sort patches according to their distance to the reference one
            table_distance.sort(key=lambda x: x[0])  # Sort by distance (first element of tuple)

            # Keep a maximum of NHW similar patches
            patch_table[k_r] = [pos for _, pos in table_distance[:nSx_r]]

            # To avoid problem
            if nSx_r == 1:
                patch_table[k_r].append(table_distance[0][1])

    patches = [(i, matches) for i, matches in enumerate(patch_table) if matches]
    stacked = []
    for i, matches in enumerate(patches):
        ref_x = matches[0] // width
        ref_y = matches[0] % width
        for match in matches[1]:
            match_x = match // width
            match_y = match % width
            a = img2d[match_x:match_x+2*kHW, match_y:match_y+2*kHW]
            if a.shape == (2*kHW, 2*kHW):
                stacked.append(a)
    patches3d = np.stack(stacked, axis=0)
    patches2d = patches3d.reshape(len(patches3d), 4 * kHW ** 2)

    return patches3d, patches2d