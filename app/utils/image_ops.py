# app/utils/image_ops.py
import numpy as np
import scipy.ndimage as ndimage

def prune_skeleton(skel, min_branch_length):
    """Lógica de poda de espolones (aislada de modelos de IA)"""
    skel = skel.copy().astype(np.uint8)
    def get_neighbors(y, x, img):
        y0, y1 = max(0, y-1), min(img.shape[0], y+2)
        x0, x1 = max(0, x-1), min(img.shape[1], x+2)
        neighborhood = img[y0:y1, x0:x1]
        indices = np.argwhere(neighborhood == 1)
        return [(y0 + i[0], x0 + i[1]) for i in indices if not (y0 + i[0] == y and x0 + i[1] == x)]

    while True:
        changed = False
        neighbor_count = ndimage.generic_filter(skel, lambda P: np.sum(P)-1 if P[4]==1 else 0, size=(3,3), mode='constant')
        endpoints = np.argwhere(neighbor_count == 1)
        for ep in endpoints:
            branch = [tuple(ep)]
            curr = tuple(ep)
            is_spur = False
            for _ in range(min_branch_length):
                neighbors = get_neighbors(curr[0], curr[1], skel)
                next_pts = [n for n in neighbors if n not in branch]
                if not next_pts or len(next_pts) > 1:
                    is_spur = True; break
                curr = next_pts[0]; branch.append(curr)
                if neighbor_count[curr[0], curr[1]] > 2:
                    is_spur = True; break
            if is_spur and len(branch) < min_branch_length:
                for y_b, x_b in branch: skel[y_b, x_b] = 0
                changed = True
        if not changed: break
    return skel