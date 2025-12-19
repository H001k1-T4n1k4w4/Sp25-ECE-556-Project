import numpy as np
from numba import jit,  prange

@jit(nopython=True, parallel=True)
def fast_block_matching(image, target_x=None, target_y=None):
    """
    Input:
        image: Grayscale of an image.
        target_x: Optional x-coordinate of target block (if None, randomly chosen)
        target_y: Optional y-coordinate of target block (if None, randomly chosen)
    
    Output:
        simlr_blocks: List of similar blocks INCLUDING the target block itself.
        simlr_indices: Pairs of indices of simlr_blocks' first pixel (upper-left corner).
        target_block: Target block that's being compared to.
        target_index: The index of the target block's first pixel (upper-left corner).
    """
    n = 8  # Block size
    N = 30  # Search window size
    
    # Height and width of the input image.
    h, w = image.shape[:2]

    if h < n or w < n:
        raise ValueError(f"Input image must be at least {n}x{n} in size.")
    
    if target_x is None:
        x = 0
    else:
        x = min(max(0, int(target_x)), h - n - 1)
        
    if target_y is None:
        y = 0
    else:
        y = min(max(0, int(target_y)), w - n - 1)

    target_block = image[x:x+n, y:y+n].copy()
    target_index = np.array([x, y], dtype=np.int32)
    
    scan_x_center = x + n//2
    scan_y_center = y + n//2
    
    scan_x_min = scan_x_center - N//2
    scan_x_max = scan_x_center + N//2
    scan_y_min = scan_y_center - N//2
    scan_y_max = scan_y_center + N//2
    
    scan_x_min = max(0, scan_x_min)
    scan_x_max = min(h - n - 1, scan_x_max)
    scan_y_min = max(0, scan_y_min)
    scan_y_max = min(w - n - 1, scan_y_max)
    
    if scan_x_min > scan_x_max or scan_y_min > scan_y_max:
        empty_blocks = np.zeros((n*n, 0), dtype=image.dtype)
        empty_indices = np.zeros((0, 2), dtype=np.int32)
        return empty_blocks, empty_indices, target_block, target_index

    t = target_block.reshape(-1)
    
    window_width = scan_x_max - scan_x_min + 1
    window_height = scan_y_max - scan_y_min + 1
    max_candidates = window_width * window_height
    
    all_distances = np.full((window_width, window_height), np.inf, dtype=np.float32)
    
    target_idx_x = x - scan_x_min
    target_idx_y = y - scan_y_min
    if 0 <= target_idx_x < window_width and 0 <= target_idx_y < window_height:
        all_distances[target_idx_x, target_idx_y] = 0.0
    
    for i in prange(scan_x_min, scan_x_max + 1):
        for j in range(scan_y_min, scan_y_max + 1):
            if (i != x or j != y) and abs(i - x) < n and abs(j - y) < n:
                continue
                
            candidate_block = image[i:i+n, j:j+n].copy()
            
            if candidate_block.shape[0] != n or candidate_block.shape[1] != n:
                continue
            
            c = candidate_block.reshape(-1)
            diff = t - c
            L2 = np.sqrt(np.sum(diff * diff))
            
            idx_x = i - scan_x_min
            idx_y = j - scan_y_min
            all_distances[idx_x, idx_y] = L2
    
    flat_distances = all_distances.flatten()
    
    sorted_indices = np.argsort(flat_distances)
    
    def index_to_coord(idx):
        y_idx = idx % window_height
        x_idx = idx // window_height
        return scan_x_min + x_idx, scan_y_min + y_idx
    
    start_idx = 0
    
    top = min(16, len(sorted_indices) - start_idx)
    
    best_blocks = np.zeros((top, n, n), dtype=image.dtype)
    best_indices = np.zeros((top, 2), dtype=np.int32)
    
    for i in range(top):
        idx = sorted_indices[i + start_idx]
        x_coord, y_coord = index_to_coord(idx)
        
        best_blocks[i] = image[x_coord:x_coord+n, y_coord:y_coord+n].copy()
        best_indices[i, 0] = x_coord
        best_indices[i, 1] = y_coord
    
    if top > 0:
        blocks_array = np.zeros((top, n*n), dtype=image.dtype)
        for i in prange(top):
            blocks_array[i] = best_blocks[i].reshape(-1)
        simlr_blocks = blocks_array.T
    else:
        simlr_blocks = np.zeros((n*n, 0), dtype=image.dtype)
    
    return simlr_blocks, best_indices, target_block, target_index