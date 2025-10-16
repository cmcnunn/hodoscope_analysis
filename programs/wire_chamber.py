import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import uproot
except ImportError:
    print("Missing dependency: uproot. Install with 'pip install uproot' and retry.")
    sys.exit(1)


# =====================================
# Configuration (overridable via env)
# =====================================
RUN_FILES_CSV = os.environ.get(
    "DRS_RUN_FILES",
    ",".join([
        "run1296_250818195045_converted.root",
        "run1264_250817143432.root"
    ]),
)

BOARD = int(os.environ.get("DRS_BOARD", 7))
GROUP_G0 = int(os.environ.get("DRS_GROUP_G0", 0))
GROUP_G1 = int(os.environ.get("DRS_GROUP_G1", 1))

# Wire chamber mapping (Group 0)
# X-plane uses channels 4 and 5; Y-plane uses channels 6 and 7
CH_X_LEFT = int(os.environ.get("CH_X_LEFT", 4))
CH_X_RIGHT = int(os.environ.get("CH_X_RIGHT", 5))
CH_Y_LOWER = int(os.environ.get("CH_Y_LOWER", 6))
CH_Y_UPPER = int(os.environ.get("CH_Y_UPPER", 7))

# Gate on Group 1 channels
# Default: include 6 (existing), plus 1 and 2 as requested
CH_GATE_G1 = int(os.environ.get("CH_GATE_G1", 6))
GATE_CHANNELS_STR = os.environ.get("DRS_GATE_CHANNELS", f"{CH_GATE_G1},1,2")
GATE_CHANNELS = []
for tok in GATE_CHANNELS_STR.split(','):
    tok = tok.strip()
    if not tok:
        continue
    try:
        val = int(tok)
        if val not in GATE_CHANNELS:
            GATE_CHANNELS.append(val)
    except ValueError:
        pass

BASELINE_SAMPLES = int(os.environ.get("BASELINE_SAMPLES", 50))
STEP_SIZE = int(os.environ.get("STEP_SIZE", 5000))

# Units and thresholds
SAMPLE_NS = float(os.environ.get("SAMPLE_NS", 0.2))  # 200 ps per sample => 0.2 ns
ADC_TO_MV = float(os.environ.get("ADC_TO_MV", 0.222))  # mV per ADC count

# Peak search window and minimum amplitude (in mV)
PEAK_MIN_NS = float(os.environ.get("PEAK_MIN_NS", 0.0))
PEAK_MAX_NS = float(os.environ.get("PEAK_MAX_NS", 200.0))
MIN_PEAK_ABS_MV = float(os.environ.get("MIN_PEAK_ABS_MV", 3.0))

# Heatmap configuration
HEATMAP_BINS = int(os.environ.get("HEATMAP_BINS", 150))
HEATMAP_RANGE_NS = float(os.environ.get("HEATMAP_RANGE_NS", 40.0))# +/- range for dx, dy (ns)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "wire_chamber_heatmap")

# Integration settings for Board1 Group1 Channel1 (match integrate_board1_g0_c0_c1.py philosophy)
INTEG_PEAK_MIN_NS = float(os.environ.get("INTEG_PEAK_MIN_NS", 0.0))
INTEG_PEAK_MAX_NS = float(os.environ.get("INTEG_PEAK_MAX_NS", 60.0))
INTEG_MIN_PEAK_ABS_MV = float(os.environ.get("INTEG_MIN_PEAK_ABS_MV", 3.0))
INTEG_RANGE_MIN_MVNS = float(os.environ.get("INTEG_RANGE_MIN_MVNS", -1000.0))
INTEG_RANGE_MAX_MVNS = float(os.environ.get("INTEG_RANGE_MAX_MVNS", 100000.0))
INTEG_SAME_TOL_MVNS = float(os.environ.get("INTEG_SAME_TOL_MVNS", 5.0))

# Board/group/channels for the all-integrals plot (ensure correct dataset)
INTEG_BOARD = int(os.environ.get("INTEG_BOARD", 7))
INTEG_GROUP = int(os.environ.get("INTEG_GROUP", 1))
INTEG_CH1 = int(os.environ.get("INTEG_CH1", 1))
INTEG_CH2 = int(os.environ.get("INTEG_CH2", 2))

# Per-channel integration windows (ns)
INTEG_C1_PEAK_MIN_NS = float(os.environ.get("INTEG_C1_PEAK_MIN_NS", 0.0))
INTEG_C1_PEAK_MAX_NS = float(os.environ.get("INTEG_C1_PEAK_MAX_NS", 60.0))
INTEG_C2_PEAK_MIN_NS = float(os.environ.get("INTEG_C2_PEAK_MIN_NS", 100.0))
INTEG_C2_PEAK_MAX_NS = float(os.environ.get("INTEG_C2_PEAK_MAX_NS", 200.0))


def resolve_tree(file) -> "uproot.TTree":
    candidates: List[str] = [
        "EventTree",
        "EventTree;34",
        "EventTree;33",
    ]
    for name in candidates:
        if name in file:
            return file[name]
    for key in file.keys():
        if str(key).startswith("EventTree"):
            return file[key]
    raise KeyError("Could not locate 'EventTree' TTree in ROOT file.")


def ns_to_sample_idx(ns_value: float) -> int:
    return int(np.clip(np.floor(ns_value / SAMPLE_NS), 0, 1023))


def baseline_to_mv(wf: np.ndarray) -> np.ndarray:
    if wf is None or not isinstance(wf, np.ndarray) or wf.size == 0:
        return np.array([], dtype=np.float64)
    baseline = float(np.mean(wf[:BASELINE_SAMPLES])) if wf.size >= BASELINE_SAMPLES else float(np.mean(wf))
    return (wf.astype(np.float64) - baseline) * ADC_TO_MV


def find_peak_index_mv(waveform_mv: np.ndarray, left_idx: int, right_idx: int, min_abs_mv: float) -> int:
    if waveform_mv is None or waveform_mv.size == 0:
        return -1
    left = max(0, int(left_idx))
    right = min(len(waveform_mv) - 1, int(right_idx))
    if right <= left:
        return -1
    segment = waveform_mv[left:right + 1]
    idx_local = int(np.argmax(np.abs(segment)))
    peak_val = float(segment[idx_local])
    if abs(peak_val) < min_abs_mv:
        return -1
    return left + idx_local


# --- Integration helpers (ported to match integrate_board1_g0_c0_c1.py) ---
def find_peak_and_window_counts(
    wf_counts: np.ndarray,
    baseline_samples: int,
    peak_min_ns: float,
    peak_max_ns: float,
    min_peak_abs_mv: float,
) -> Tuple[int, float, int, int]:
    """
    Returns (peak_index, peak_value_counts, left_idx, right_idx) where left/right are
    10%-of-peak absolute threshold crossings around the extremum, following the same
    approach used in integrate_board1_g0_c0_c1.py. Threshold check uses mV.
    """
    if wf_counts is None or wf_counts.size == 0:
        return -1, 0.0, -1, -1

    baseline = float(np.mean(wf_counts[:baseline_samples])) if wf_counts.size >= baseline_samples else float(np.mean(wf_counts))
    w = wf_counts.astype(np.float64) - baseline

    start_idx = max(0, int(peak_min_ns / SAMPLE_NS))
    end_idx = min(w.size - 1, int(peak_max_ns / SAMPLE_NS))
    if end_idx <= start_idx:
        return -1, 0.0, -1, -1
    w_window = w[start_idx:end_idx + 1]
    if w_window.size == 0:
        return -1, 0.0, -1, -1
    local_peak = int(np.argmax(np.abs(w_window)))
    peak_idx = start_idx + local_peak
    peak_val_counts = float(w[peak_idx])
    peak_abs_mv = abs(peak_val_counts) * ADC_TO_MV
    if peak_abs_mv < min_peak_abs_mv:
        return -1, 0.0, -1, -1

    thr = 0.1 * abs(peak_val_counts)

    # left crossing
    left_idx = -1
    left_search_start = start_idx
    left_search_end = max(peak_idx, left_search_start + 1)
    for i in range(left_search_start, left_search_end - 1):
        if abs(w[i]) < thr and abs(w[i + 1]) >= thr:
            left_idx = i + 1
            break
    if left_idx == -1:
        left_idx = left_search_start if abs(w[left_search_start]) >= thr else -1

    # right crossing
    right_idx = -1
    right_search_end = end_idx
    for i in range(peak_idx, right_search_end):
        if abs(w[i]) >= thr and abs(w[i + 1]) < thr:
            right_idx = i + 1
            break
    if right_idx == -1:
        right_idx = right_search_end if abs(w[right_search_end]) >= thr else -1

    if left_idx == -1 or right_idx == -1 or right_idx <= left_idx:
        return -1, 0.0, -1, -1

    return peak_idx, peak_val_counts, left_idx, right_idx


def integrate_window_counts(wf_counts: np.ndarray, left: int, right: int, baseline_samples: int) -> float:
    if wf_counts is None or left < 0 or right < 0 or right <= left:
        return float("nan")
    baseline = float(np.mean(wf_counts[:baseline_samples])) if wf_counts.size >= baseline_samples else float(np.mean(wf_counts))
    w = wf_counts.astype(np.float64) - baseline
    segment = w[left:right + 1]
    if segment.size < 2:
        return float("nan")
    return float(np.trapz(segment, dx=1.0))

def integrate_fixed_window_counts(wf_counts: np.ndarray, start_ns: float, end_ns: float, baseline_samples: int) -> float:
    if wf_counts is None or wf_counts.size == 0:
        return float("nan")
    left = max(0, int(start_ns / SAMPLE_NS))
    right = min(int(end_ns / SAMPLE_NS), wf_counts.size - 1)
    if right <= left:
        return float("nan")
    baseline = float(np.mean(wf_counts[:baseline_samples])) if wf_counts.size >= baseline_samples else float(np.mean(wf_counts))
    w = wf_counts.astype(np.float64) - baseline
    segment = w[left:right + 1]
    if segment.size < 2:
        return float("nan")
    return float(np.trapz(segment, dx=1.0))


def main():
    run_files = [s.strip() for s in RUN_FILES_CSV.split(',') if s.strip()]
    if not run_files:
        print("No ROOT files specified.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Branch names (group 0 positions)
    b_x_l = f"DRS_Board{BOARD}_Group{GROUP_G0}_Channel{CH_X_LEFT}"
    b_x_r = f"DRS_Board{BOARD}_Group{GROUP_G0}_Channel{CH_X_RIGHT}"
    b_y_l = f"DRS_Board{BOARD}_Group{GROUP_G0}_Channel{CH_Y_LOWER}"
    b_y_u = f"DRS_Board{BOARD}_Group{GROUP_G0}_Channel{CH_Y_UPPER}"

    left_idx = ns_to_sample_idx(PEAK_MIN_NS)
    right_idx = ns_to_sample_idx(PEAK_MAX_NS)

    # Accumulate results per gate channel across all runs
    dx_by_gate = {gate_ch: [] for gate_ch in GATE_CHANNELS}
    dy_by_gate = {gate_ch: [] for gate_ch in GATE_CHANNELS}

    for run_path in run_files:
        if not os.path.exists(run_path):
            print(f"Warning: file not found, skipping: {run_path}")
            continue

        try:
            f = uproot.open(run_path)
            tree = resolve_tree(f)
        except Exception as e:
            print(f"Failed to open/locate tree in {run_path}: {e}")
            continue

        available = set(tree.keys())
        required_g0 = [b_x_l, b_x_r, b_y_l, b_y_u]
        if not all(b in available for b in required_g0):
            print(f"Skipping {os.path.basename(run_path)}: not all G0 channels present.")
            continue

        # Process each requested gate channel separately
        for gate_ch in GATE_CHANNELS:
            b_gate = f"DRS_Board{BOARD}_Group{GROUP_G1}_Channel{gate_ch}"
            if b_gate not in available:
                print(f"Warning: gate branch missing in {os.path.basename(run_path)}: {b_gate}")
                continue

            expressions = required_g0 + [b_gate]
            print(f"Processing: {os.path.basename(run_path)} with gate {b_gate}")

            try:
                for arrays in tree.iterate(expressions=expressions, library="np", step_size=STEP_SIZE):
                    n_in_chunk = len(arrays[required_g0[0]]) if required_g0[0] in arrays else 0
                    for i_evt in range(n_in_chunk):
                        wf_xl = arrays[b_x_l][i_evt]
                        wf_xr = arrays[b_x_r][i_evt]
                        wf_yl = arrays[b_y_l][i_evt]
                        wf_yu = arrays[b_y_u][i_evt]

                        # Baseline-subtracted waveforms in mV
                        w_xl = baseline_to_mv(wf_xl)
                        w_xr = baseline_to_mv(wf_xr)
                        w_yl = baseline_to_mv(wf_yl)
                        w_yu = baseline_to_mv(wf_yu)

                        # Peak indices
                        p_xl = find_peak_index_mv(w_xl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_xr = find_peak_index_mv(w_xr, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_yl = find_peak_index_mv(w_yl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_yu = find_peak_index_mv(w_yu, left_idx, right_idx, MIN_PEAK_ABS_MV)

                        if p_xl < 0 or p_xr < 0 or p_yl < 0 or p_yu < 0:
                            continue

                        # Gate condition: require valid peak in the chosen Group1 channel
                        wf_gate = arrays[b_gate][i_evt]
                        w_gate = baseline_to_mv(wf_gate)
                        p_gate = find_peak_index_mv(w_gate, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        if p_gate < 0:
                            continue

                        # Position proxies as time differences (ns)
                        dx_ns = (p_xr - p_xl) * SAMPLE_NS
                        dy_ns = (p_yu - p_yl) * SAMPLE_NS

                        if abs(dx_ns) <= 5 * HEATMAP_RANGE_NS and abs(dy_ns) <= 5 * HEATMAP_RANGE_NS:
                            dx_by_gate[gate_ch].append(dx_ns)
                            dy_by_gate[gate_ch].append(dy_ns)
            except Exception as e:
                print(f"Iteration error in {os.path.basename(run_path)} (gate {gate_ch}): {e}")

    # Output per-gate CSV and heatmap
    rng = HEATMAP_RANGE_NS
    x_range = (-rng, rng)
    y_range = (-rng, rng)

    for gate_ch in GATE_CHANNELS:
        dx_list = dx_by_gate.get(gate_ch, [])
        dy_list = dy_by_gate.get(gate_ch, [])
        if not dx_list:
            print(f"No gated events for Group{GROUP_G1} Channel{gate_ch}; skipping outputs.")
            continue

        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, f"wire_chamber_positions_ns_g1_ch{gate_ch}.csv")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            import csv
            with open(csv_path, mode="w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["dx_ns", "dy_ns"])  # time differences in ns
                for x, y in zip(dx_list, dy_list):
                    writer.writerow([float(x), float(y)])
            print(f"Saved: {csv_path} ({len(dx_list)} rows)")
        except Exception as e:
            print(f"Failed to write CSV {csv_path}: {e}")

        # Heatmap
        dx = np.array(dx_list, dtype=np.float64)
        dy = np.array(dy_list, dtype=np.float64)
        plt.figure(figsize=(8, 7))
        plt.hist2d(dx, dy, bins=HEATMAP_BINS, range=[x_range, y_range], cmap="viridis")
        plt.xlabel("X time difference: Channel5 − Channel4 (ns)")
        plt.ylabel("Y time difference: Channel7 − Channel6 (ns)")
        plt.title(f"Wire chamber hit map — gated on Board{BOARD} Group{GROUP_G1} Channel{gate_ch}")
        plt.colorbar(label="Counts")
        plt.grid(False)
        plt.tight_layout()
        outpng = os.path.join(OUTPUT_DIR, f"wire_chamber_heatmap_board{BOARD}_g1_ch{gate_ch}.png")
        plt.savefig(outpng, dpi=180)
        plt.close()
        print(f"Saved: {outpng}")

    # Additional positional heatmaps: gate on Board{BOARD} Group 2 channels 0 and 1
    for g2_gate_ch in [0, 1]:
        dx_list_g2: List[float] = []
        dy_list_g2: List[float] = []
        for run_path in run_files:
            if not os.path.exists(run_path):
                continue
            try:
                f = uproot.open(run_path)
                tree = resolve_tree(f)
            except Exception:
                continue
            available = set(tree.keys())
            required_g0 = [b_x_l, b_x_r, b_y_l, b_y_u]
            b_g2_gate = f"DRS_Board{BOARD}_Group{2}_Channel{g2_gate_ch}"
            if not all(b in available for b in required_g0) or b_g2_gate not in available:
                continue
            expressions = required_g0 + [b_g2_gate]
            try:
                for arrays in tree.iterate(expressions=expressions, library="np", step_size=STEP_SIZE):
                    n_in_chunk = len(arrays[required_g0[0]]) if required_g0[0] in arrays else 0
                    for i_evt in range(n_in_chunk):
                        wf_xl = arrays[b_x_l][i_evt]
                        wf_xr = arrays[b_x_r][i_evt]
                        wf_yl = arrays[b_y_l][i_evt]
                        wf_yu = arrays[b_y_u][i_evt]
                        # Baseline-subtracted waveforms in mV
                        w_xl = baseline_to_mv(wf_xl)
                        w_xr = baseline_to_mv(wf_xr)
                        w_yl = baseline_to_mv(wf_yl)
                        w_yu = baseline_to_mv(wf_yu)
                        # Peak indices in WC
                        p_xl = find_peak_index_mv(w_xl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_xr = find_peak_index_mv(w_xr, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_yl = find_peak_index_mv(w_yl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        p_yu = find_peak_index_mv(w_yu, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        if p_xl < 0 or p_xr < 0 or p_yl < 0 or p_yu < 0:
                            continue
                        # Gate on Group 2 channel
                        wf_g2 = arrays[b_g2_gate][i_evt]
                        w_g2 = baseline_to_mv(wf_g2)
                        p_g2 = find_peak_index_mv(w_g2, left_idx, right_idx, MIN_PEAK_ABS_MV)
                        if p_g2 < 0:
                            continue
                        dx_ns = (p_xr - p_xl) * SAMPLE_NS
                        dy_ns = (p_yu - p_yl) * SAMPLE_NS
                        if abs(dx_ns) <= 5 * HEATMAP_RANGE_NS and abs(dy_ns) <= 5 * HEATMAP_RANGE_NS:
                            dx_list_g2.append(dx_ns)
                            dy_list_g2.append(dy_ns)
            except Exception:
                continue
        if dx_list_g2:
            # CSV
            csv_path = os.path.join(OUTPUT_DIR, f"wire_chamber_positions_ns_gatedBy_g2_ch{g2_gate_ch}.csv")
            try:
                import csv
                with open(csv_path, mode="w", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(["dx_ns", "dy_ns"])
                    for x, y in zip(dx_list_g2, dy_list_g2):
                        writer.writerow([float(x), float(y)])
                print(f"Saved: {csv_path} ({len(dx_list_g2)} rows)")
            except Exception as e:
                print(f"Failed to write CSV {csv_path}: {e}")
            # Heatmap
            dx = np.array(dx_list_g2, dtype=np.float64)
            dy = np.array(dy_list_g2, dtype=np.float64)
            plt.figure(figsize=(8, 7))
            plt.hist2d(dx, dy, bins=HEATMAP_BINS, range=[x_range, y_range], cmap="viridis")
            plt.xlabel("X time difference: Channel5 − Channel4 (ns)")
            plt.ylabel("Y time difference: Channel7 − Channel6 (ns)")
            plt.title(f"WC map — gated on G2 Ch{g2_gate_ch}")
            plt.colorbar(label="Counts")
            plt.grid(False)
            plt.tight_layout()
            outpng = os.path.join(OUTPUT_DIR, f"wire_chamber_heatmap_board{BOARD}_gatedBy_g2_ch{g2_gate_ch}.png")
            plt.savefig(outpng, dpi=180)
            plt.close()
            print(f"Saved: {outpng}")

    # --- New plot: match events where B1G1C1 and B1G1C2 integrals are the same (within tolerance) ---
    b1g1c1_branch = f"DRS_Board{1}_Group{1}_Channel{1}"
    b1g1c2_branch = f"DRS_Board{1}_Group{1}_Channel{2}"

    dx_match: List[float] = []
    dy_match: List[float] = []
    integ1_vals: List[float] = []
    integ2_vals: List[float] = []
    # Collect all integrals (regardless of matching) for no-position heatmap
    integ1_all: List[float] = []
    integ2_all: List[float] = []

    for run_path in run_files:
        if not os.path.exists(run_path):
            continue
        try:
            f = uproot.open(run_path)
            tree = resolve_tree(f)
        except Exception:
            continue

        available = set(tree.keys())
        required_g0 = [b_x_l, b_x_r, b_y_l, b_y_u]
        if not all(b in available for b in required_g0):
            continue
        if b1g1c1_branch not in available or b1g1c2_branch not in available:
            print(f"Warning: missing B1G1C1/C2 in {os.path.basename(run_path)}; skipping for same-integral heatmap")
            continue

        expressions = required_g0 + [b1g1c1_branch, b1g1c2_branch]
        try:
            for arrays in tree.iterate(expressions=expressions, library="np", step_size=STEP_SIZE):
                n_in_chunk = len(arrays[required_g0[0]]) if required_g0[0] in arrays else 0
                for i_evt in range(n_in_chunk):
                    # Positions
                    wf_xl = arrays[b_x_l][i_evt]
                    wf_xr = arrays[b_x_r][i_evt]
                    wf_yl = arrays[b_y_l][i_evt]
                    wf_yu = arrays[b_y_u][i_evt]

                    w_xl = baseline_to_mv(wf_xl)
                    w_xr = baseline_to_mv(wf_xr)
                    w_yl = baseline_to_mv(wf_yl)
                    w_yu = baseline_to_mv(wf_yu)

                    p_xl = find_peak_index_mv(w_xl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                    p_xr = find_peak_index_mv(w_xr, left_idx, right_idx, MIN_PEAK_ABS_MV)
                    p_yl = find_peak_index_mv(w_yl, left_idx, right_idx, MIN_PEAK_ABS_MV)
                    p_yu = find_peak_index_mv(w_yu, left_idx, right_idx, MIN_PEAK_ABS_MV)
                    if p_xl < 0 or p_xr < 0 or p_yl < 0 or p_yu < 0:
                        continue

                    dx_ns = (p_xr - p_xl) * SAMPLE_NS
                    dy_ns = (p_yu - p_yl) * SAMPLE_NS
                    if not (abs(dx_ns) <= 5 * HEATMAP_RANGE_NS and abs(dy_ns) <= 5 * HEATMAP_RANGE_NS):
                        continue

                    # Integrals for B1G1C1 and B1G1C2
                    wf_c1 = arrays[b1g1c1_branch][i_evt]
                    wf_c2 = arrays[b1g1c2_branch][i_evt]

                    _, _, l1, r1 = find_peak_and_window_counts(wf_c1, BASELINE_SAMPLES, INTEG_C1_PEAK_MIN_NS, INTEG_C1_PEAK_MAX_NS, INTEG_MIN_PEAK_ABS_MV)
                    _, _, l2, r2 = find_peak_and_window_counts(wf_c2, BASELINE_SAMPLES, INTEG_C2_PEAK_MIN_NS, INTEG_C2_PEAK_MAX_NS, INTEG_MIN_PEAK_ABS_MV)
                    if l1 == -1 or r1 == -1 or l2 == -1 or r2 == -1:
                        continue

                    area1 = integrate_window_counts(wf_c1, l1, r1, BASELINE_SAMPLES)
                    area2 = integrate_window_counts(wf_c2, l2, r2, BASELINE_SAMPLES)
                    if np.isnan(area1) or np.isnan(area2):
                        continue
                    integ1 = -area1 * ADC_TO_MV * SAMPLE_NS
                    integ2 = -area2 * ADC_TO_MV * SAMPLE_NS

                    if abs(integ1 - integ2) <= INTEG_SAME_TOL_MVNS:
                        dx_match.append(dx_ns)
                        dy_match.append(dy_ns)
                        integ1_vals.append(float(integ1))
                        integ2_vals.append(float(integ2))
        except Exception as e:
            print(f"Iteration error for same-integral heatmap in {os.path.basename(run_path)}: {e}")

    if dx_match:
        rng = HEATMAP_RANGE_NS
        x_range = (-rng, rng)
        y_range = (-rng, rng)

        plt.figure(figsize=(8, 7))
        plt.hist2d(np.array(dx_match), np.array(dy_match), bins=HEATMAP_BINS, range=[x_range, y_range], cmap="magma")
        plt.xlabel("X time difference: Channel5 − Channel4 (ns)")
        plt.ylabel("Y time difference: Channel7 − Channel6 (ns)")
        plt.title(f"WC positions where B1G1C1 and C2 integrals match (≤ {INTEG_SAME_TOL_MVNS} mV·ns)")
        plt.colorbar(label="Counts")
        plt.grid(False)
        plt.tight_layout()
        outpng = os.path.join(OUTPUT_DIR, "wire_chamber_heatmap_sameIntegral_B1G1C1_C2.png")
        plt.savefig(outpng, dpi=180)
        plt.close()
        print(f"Saved: {outpng}")

        # CSV of matched events
        import csv
        csv_path = os.path.join(OUTPUT_DIR, "wc_positions_b1g1c1_c2_same_integral.csv")
        try:
            with open(csv_path, mode="w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["dx_ns", "dy_ns", "b1g1c1_integral_mVns", "b1g1c2_integral_mVns"])
                for x, y, v1, v2 in zip(dx_match, dy_match, integ1_vals, integ2_vals):
                    writer.writerow([float(x), float(y), float(v1), float(v2)])
            print(f"Saved: {csv_path} ({len(dx_match)} rows)")
        except Exception as e:
            print(f"Failed to write same-integral CSV: {e}")

    # Compute ALL integrals for Board7 Group1 Channels 1 & 2 (no position gating) and plot heatmap
    b_all_c1 = f"DRS_Board{INTEG_BOARD}_Group{INTEG_GROUP}_Channel{INTEG_CH1}"
    b_all_c2 = f"DRS_Board{INTEG_BOARD}_Group{INTEG_GROUP}_Channel{INTEG_CH2}"
    integ1_all = []
    integ2_all = []
    for run_path in run_files:
        if not os.path.exists(run_path):
            continue
        try:
            f = uproot.open(run_path)
            tree = resolve_tree(f)
        except Exception:
            continue
        available = set(tree.keys())
        if b_all_c1 not in available or b_all_c2 not in available:
            print(f"Warning: missing {b_all_c1} or {b_all_c2} in {os.path.basename(run_path)}; skipping in all-integrals heatmap")
            continue
        try:
            for arrays in tree.iterate(expressions=[b_all_c1, b_all_c2], library="np", step_size=STEP_SIZE):
                n_in_chunk = len(arrays[b_all_c1]) if b_all_c1 in arrays else 0
                for i_evt in range(n_in_chunk):
                    wf_c1 = arrays[b_all_c1][i_evt]
                    wf_c2 = arrays[b_all_c2][i_evt]
                    # Try 10% crossing window; fallback to fixed [INTEG_PEAK_MIN_NS, INTEG_PEAK_MAX_NS]
                    l1 = r1 = l2 = r2 = -1
                    p1 = find_peak_and_window_counts(wf_c1, BASELINE_SAMPLES, INTEG_PEAK_MIN_NS, INTEG_PEAK_MAX_NS, INTEG_MIN_PEAK_ABS_MV)
                    p2 = find_peak_and_window_counts(wf_c2, BASELINE_SAMPLES, INTEG_PEAK_MIN_NS, INTEG_PEAK_MAX_NS, INTEG_MIN_PEAK_ABS_MV)
                    if p1 is not None:
                        _, _, l1, r1 = p1
                    if p2 is not None:
                        _, _, l2, r2 = p2
                    if l1 != -1 and r1 != -1:
                        area1 = integrate_window_counts(wf_c1, l1, r1, BASELINE_SAMPLES)
                    else:
                        area1 = integrate_fixed_window_counts(wf_c1, INTEG_C1_PEAK_MIN_NS, INTEG_C1_PEAK_MAX_NS, BASELINE_SAMPLES)
                    if l2 != -1 and r2 != -1:
                        area2 = integrate_window_counts(wf_c2, l2, r2, BASELINE_SAMPLES)
                    else:
                        area2 = integrate_fixed_window_counts(wf_c2, INTEG_C2_PEAK_MIN_NS, INTEG_C2_PEAK_MAX_NS, BASELINE_SAMPLES)
                    if np.isnan(area1) or np.isnan(area2):
                        continue
                    integ1_all.append(float(-area1 * ADC_TO_MV * SAMPLE_NS))
                    integ2_all.append(float(-area2 * ADC_TO_MV * SAMPLE_NS))
        except Exception as e:
            print(f"Iteration error for all-integrals heatmap in {os.path.basename(run_path)}: {e}")

    if integ1_all and integ2_all:
        v1 = np.array(integ1_all, dtype=np.float64)
        v2 = np.array(integ2_all, dtype=np.float64)
        # Debug stats to verify dynamic range (all events, no WC gating)
        try:
            def q(a, p):
                return float(np.percentile(a, p))
            print(
                "ALL-INTEGRALS stats — C1:",
                "n=", len(v1),
                "min=", float(np.min(v1)),
                "p50=", q(v1, 50),
                "p95=", q(v1, 95),
                "max=", float(np.max(v1)),
                "| C2:",
                "n=", len(v2),
                "min=", float(np.min(v2)),
                "p50=", q(v2, 50),
                "p95=", q(v2, 95),
                "max=", float(np.max(v2)),
            )
        except Exception:
            pass
        v_min = float(min(v1.min(), v2.min()))
        v_max = float(max(v1.max(), v2.max()))
        pad = 0.05 * (v_max - v_min) if v_max > v_min else 1.0
        r = (v_min - pad, v_max + pad)

        plt.figure(figsize=(8, 7))
        # Compute 2D histogram to control zero-count display
        counts, xedges, yedges = np.histogram2d(v1, v2, bins=1000, range=[r, r])
        masked = np.ma.masked_where(counts.T == 0, counts.T)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        mesh = plt.pcolormesh(xedges, yedges, masked, cmap=cmap, shading="auto")
        plt.xlabel(f"Board{INTEG_BOARD} Group{INTEG_GROUP} Channel{INTEG_CH1} integral (mV·ns)")
        plt.ylabel(f"Board{INTEG_BOARD} Group{INTEG_GROUP} Channel{INTEG_CH2} integral (mV·ns)")
        plt.title(f"Board{INTEG_BOARD} Group{INTEG_GROUP} Channel{INTEG_CH1} vs Channel{INTEG_CH2} integrals (all events)")
        plt.colorbar(mesh, label="Counts")
        plt.xlim(0, 200)
        plt.ylim(0, 200)
        plt.grid(False)
        plt.tight_layout()
        outpng2 = os.path.join(OUTPUT_DIR, "integrals_heatmap_B1G1C1_vs_C2_all.png")
        plt.savefig(outpng2, dpi=180)
        plt.close()
        print(f"Saved: {outpng2}")


if __name__ == "__main__":
    main()