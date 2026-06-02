import numpy as np

def align_for_ml_robust(s1_win, s2_win, freq, tolerance):
    """
    Refactored for High-Volume Data: Uses tolerance matching instead of hard grids.
    """
    dt = 1000 / freq   # 40ms

    # 2. Use the sensor with fewer rows as the "Anchor"
    if len(s1_win) < len(s2_win):
        anchor, to_align = s1_win, s2_win
        is_leg_anchor = True
    else:
        anchor, to_align = s2_win, s1_win
        is_leg_anchor = False

    # 3. Find the closest match in 'to_align' for every timestamp in 'anchor'
    # searchsorted is extremely fast for 15 million rows
    indices = np.searchsorted(to_align[:, 0], anchor[:, 0])

    # Handle edge cases where searchsorted goes out of bounds
    indices = np.clip(indices, 0, len(to_align) - 1)

    # 4. Critical: Tolerance Check
    # Only keep matches where the time difference is less than a frequece with a tolerance
    time_diffs = np.abs(to_align[indices, 0] - anchor[:, 0])
    valid_mask = time_diffs < (dt * tolerance )

    final_anchor = anchor[valid_mask]
    final_aligned = to_align[indices[valid_mask]]

    if is_leg_anchor:
        return final_anchor, final_aligned
    else:
        return final_aligned, final_anchor

# Mock data
s_PI = np.array([[1010,11,11], [2020,12,12], [3030,13,13], [5040,15,25], [6050,16,16]])
s_M = np.array([[1001, 21, 21], [2002,22,22], [3003,23,23], [4004,24,24]])

## align not structured datasets
(s1_sync, s2_sync) = align_for_ml_robust(s_PI, s_M, 1, 0.05)

## results
print(s1_sync)
print(s2_sync)
