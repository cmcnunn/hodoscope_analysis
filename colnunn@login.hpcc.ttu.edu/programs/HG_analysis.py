import ROOT
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import csv
import json
import time

# ==============================
# GLOBAL CONFIGURATION
# ==============================
BOARD1 = "FERS_Board0_energyHG" #X-axis 
BOARD2 = "FERS_Board1_energyHG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6" #VETO Measuments 
MCP1 = "DRS_Board0_Group3_Channel6" #MCP1 Measuments
MCP2 = "DRS_Board0_Group3_Channel7" #MCP2 Measuments
THRESHOLD = 5500  # Minimum ADC value for HIGH GAIN event selection

n_entries = 0
run_number = "0000"
tree = None
file = None
FILE_PATH = ""
RUN_DIR = ""
SAVE_DIR_EVENTS = ""
SAVE_DIR_FULL = ""
SAVE_DIR_CHANNEL = ""
HITS_CSV = ""

# ==============================
# CHANNEL MAPPING CONFIGURATIONS
# ==============================

#SNAKE MAPPING + FERS MAPPING 
s_fers_mapping = [
     0, 8,16,24,32,40,48,56,
    60,52,44,36,28,20,12, 4,
     2,10,18,26,34,42,50,58,
    62,54,46,38,30,22,14, 6,
     1, 9,17,25,33,41,49,57,
    61,53,45,37,29,21,13, 5,
     3,11,19,27,35,43,51,59,
    63,55,47,39,31,23,15, 7
]


test_mapping = [
    63, 55, 47, 39, 31, 23, 15, 7,
    3, 11, 19, 27, 35, 43, 51, 59,
    61, 53, 45, 37, 29, 21, 13, 5,
    1, 9, 17, 25, 33, 41, 49, 57,
    62, 54, 46, 38, 30, 22, 14, 6,
    2, 10, 18, 26, 34, 42, 50, 58,
    60, 52, 44, 36, 28, 20, 12, 4,
    0, 8, 16, 24, 32, 40, 48, 56
]
######CHANGE HERE######
mapping1 = s_fers_mapping
mapping2 = test_mapping
#######################
def do_map(data, apply_mapping):
    """Apply channel remapping to a list of 64 ADC values."""
    if len(data) != len(apply_mapping):
        raise ValueError("Data length and mapping length must match.")
    remapped_data = [data[i] for i in apply_mapping]
    return remapped_data

def getupstreamVeto(entry, veto_board=BOARD3, veto_threshold=13.6):
    """Check the veto for one event"""
    tree.GetEntry(entry)
    veto_energy = list(getattr(tree,veto_board))
    veto_energy = np.min(veto_energy)
    return veto_energy < veto_threshold

def process_event(entry,remap=True, board1=BOARD1, board2=BOARD2, threshold=THRESHOLD):
    tree.GetEntry(entry)
    energies_1 = list(getattr(tree, board1))
    energies_2 = list(getattr(tree, board2))
    if remap:
        energies_1 = do_map(energies_1,mapping1)
        energies_2 = do_map(energies_2,mapping2)
    all_channels = energies_1 + energies_2
    hits = [i for i, val in enumerate(all_channels) if val > threshold]
    multiplicity = len(hits)
    # compute mean ADC of all channels above threshold (if any)
    mean_adc = np.mean([val for val in all_channels if val > threshold]) if hits else 0
    return energies_1, energies_2, multiplicity, mean_adc

def detect_hits(entry,map=False, board1=BOARD1, board2=BOARD2, threshold=THRESHOLD):
    """Detect hits in both boards for a given event entry."""
    energies_1, energies_2, _, _ = process_event(entry, remap=map, board1=board1, board2=board2, threshold=threshold)
    hits1 = [i for i, val in enumerate(energies_2) if val > threshold]
    hits2 = [i for i, val in enumerate(energies_1) if val > threshold]
    return hits1, hits2

def get_runs(json_file):
    with open(json_file, "r") as f:
        runs = json.load(f)
    run = input(f"Enter run number to analyze (or 'all' for all runs): ")
    run_info = []
    if run == 'all':
        for run_number, info in runs.items():
            file_path = info.get("file")
            if file_path:
                run_info.append((file_path, run_number))
        else:
            print(f"Warning: Run {run_number} missing file path")
    elif run != 'all':
        info = runs.get(run)
        if info:
            file_path = info.get("file")
            if file_path:
                run_info.append((file_path, run))
            else:
                print(f"Error: Run {run} missing file path")
        else:
            print(f"Error: Run {run} not found in {json_file}")
    return run_info 


def analyze_run(events = n_entries, max_hits=2, remap=True, board1=BOARD1, board2=BOARD2, board3=BOARD3, threshold=THRESHOLD, veto_threshold=13.6):
    """Analyze a full run of events."""
    events = range(events) if isinstance(events, int) else events
    hitx = []
    hity = []
    multiplicites = []
    mean_adcs = []
    hodoscope_events = []
    hit_detection_events = []

    #Save hits to CSV file
    with open(HITS_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["EventID", "Multiplicity", "Hits_Board1", "Hits_Board2", "Mean_ADC"])
        for event_id in events:
            if getupstreamVeto(event_id, veto_board=board3, veto_threshold=veto_threshold):
                _, _, multiplicity, mean_adc = process_event(event_id, remap=remap, board1=board1, board2=board2, threshold=threshold)
                hits1, hits2 = detect_hits(event_id, map=remap, board1=board1, board2=board2, threshold=threshold)
                hodoscope_events.append(event_id)
                if len(hits1) < max_hits or len(hits2) < max_hits and len(hits1) > 0 and len(hits2) > 0:
                    hit_detection_events.append(event_id)
                    hitx.extend(hits1)
                    hity.extend(hits2)
                    multiplicites.append(multiplicity)
                    mean_adcs.append(mean_adc)
                    writer.writerow([event_id,multiplicity, hits1, hits2, mean_adc])
                else:
                    continue
            else:
                continue
    
    print(f"[INFO] Total events processed: {len(events)}")
    print(f"[INFO] Events passing veto: {len(hodoscope_events)}")
    print(f"[INFO] Events with hits detected: {len(hit_detection_events)}")

def check_run():
    a = input(f"[WARNING] {HITS_CSV} already exists. Do you want to re-analyze the run and overwrite the existing data? (y/n): ")
    if a.lower() == 'y':
        print(f"[INFO] Overwriting existing data in {HITS_CSV}.")
        analyze_run(events=n_entries, max_hits=2, remap=True, board1=BOARD1, board2=BOARD2, board3=BOARD3, threshold=THRESHOLD, veto_threshold=13.6)
    else:
        print(f"[INFO] Skipping analysis for this run.")

def plot_run(csv_file, title=f"Run {run_number} Hit Map"):
    """
    Plot the hit map by reading hit data from the CSV file.
    Handles cases where list columns are sometimes quoted and sometimes not.
    """
    hitx, hity = [], []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize weird entries like [] or quoted lists
            x_raw, y_raw = row["Hits_Board1"], row["Hits_Board2"]

            if x_raw.strip() == "[]":
                xlist = []
            else:
                xlist = ast.literal_eval(x_raw)

            if y_raw.strip() == "[]":
                ylist = []
            else:
                ylist = ast.literal_eval(y_raw)

            hitx.extend(xlist)
            hity.extend(ylist)

    # Build the 64x64 hitmap
    Z = np.zeros((64, 64))
    for x, y in zip(hitx, hity):
        if 0 <= x < 64 and 0 <= y < 64:
            Z[y, x] += 1

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin="lower", cmap="viridis")
    plt.colorbar(label="Counts")
    plt.xlabel("Board 1 Channels")
    plt.ylabel("Board 2 Channels")
    plt.title(title)
    plt.savefig(os.path.join(SAVE_DIR_FULL, f"{title.replace(' ', '_')}.png"), dpi=300)
    plt.close()

def profile2d(x, y, z, xedges=None, yedges=None, bins=(64, 64),
              xr=None, yr=None, min_count=1):
    """
    TProfile2D-like reducer: mean(z) in each (x,y) bin.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)

    # Drop non-finite rows
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    if xedges is None or yedges is None:
        if xr is None: xr = (np.nanmin(x), np.nanmax(x))
        if yr is None: yr = (np.nanmin(y), np.nanmax(y))
        xedges = np.linspace(xr[0], xr[1], bins[0] + 1)
        yedges = np.linspace(yr[0], yr[1], bins[1] + 1)
    xedges = np.asarray(xedges); yedges = np.asarray(yedges)

    # counts, sum(z), sum(z^2)
    count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    zsum,  _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=z)
    z2sum, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=z*z)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = zsum / count
        var  = z2sum / np.maximum(count, 1) - mean**2
        var[var < 0] = 0.0
        stderr = np.sqrt(var) / np.sqrt(np.maximum(count, 1))

    mask = count < min_count
    mean = mean.astype(float)
    stderr = stderr.astype(float)
    mean[mask] = np.nan
    stderr[mask] = np.nan

    return mean, count, xedges, yedges, stderr

def veto_profile2d(events, tree, veto_board="FERS_Board2_energyHG", 
                   bins=(64, 64), xr=(0, 64), yr=(0, 64),
                   min_count=1, title=f"{run_number} Cumulative Veto Amplitude (mean)"):
    """
    Build a TProfile2D-like heatmap of veto amplitude vs (hit_x, hit_y).

    Parameters
    ----------
    events : iterable of int
        Event IDs to process
    tree : ROOT TTree
        Event tree
    veto_board : str
        Branch name for veto amplitudes
    bins : (nx, ny)
        Number of bins in (x,y)
    xr, yr : tuple
        Range for x and y axes
    min_count : int
        Minimum entries per bin to include mean
    title : str
        Plot title
    """

    # --- Gather data ---
    hit_xs, hit_ys, veto_amps = [], [], []

    with open(HITS_CSV, "r") as f:
        reader = csv.DictReader(f)
        hit_data = {int(row["EventID"]): (ast.literal_eval(row["Hits_Board1"]), ast.literal_eval(row["Hits_Board2"])) for row in reader}
    for event_id in events:
        tree.GetEntry(event_id)
        veto_amp = np.min(list(getattr(tree, veto_board)))
        hits_x, hits_y = hit_data.get(event_id, ([], []))
        if not hits_x or not hits_y:
            continue  # skip events with no hits

        # expand multiple hits
        for x in hits_x:
            for y in hits_y:
                hit_xs.append(x)
                hit_ys.append(y)
                veto_amps.append(veto_amp)

    hit_xs, hit_ys, veto_amps = map(np.array, (hit_xs, hit_ys, veto_amps))

    # --- Profile2D reduction ---
    mean, count, xedges, yedges, stderr = profile2d(
        hit_xs, hit_ys, veto_amps,
        bins=bins, xr=xr, yr=yr, min_count=min_count
    )

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    pcm = plt.pcolormesh(xedges, yedges, mean.T, shading="auto")
    plt.colorbar(pcm, label="Mean Veto Amplitude (ADC)")
    plt.xlabel("Board 1 Channels (X-axis)")
    plt.ylabel("Board 2 Channels (Y-axis)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_FULL, f"{title.replace(' ', '_')}.png"), dpi=300)
    plt.close()
    return mean, count, xedges, yedges, stderr

def plot_mean_adc_histogram(csv_file, title=f"Run {run_number} Mean ADC Histogram"):
    mean_adc =[]
    with open(csv_file, "r") as f: 
        reader = csv.DictReader(f)
        for row in reader:
            adc_row  = row["Mean_ADC"]
            if adc_row.strip() == "0":
                continue
            else:
                mean_adc.append(float(adc_row))
    mean_adc = np.array(mean_adc)
    mean_val = np.mean(mean_adc)
    std_val = np.std(mean_adc)
    count_val = len(mean_adc)

    plt.figure(figsize=(8,6))
    plt.hist(mean_adc, bins=50, color='blue', alpha=0.7, density=True)
    plt.xlabel("Mean ADC Value")
    plt.ylabel("Density")
    stats_label = f"Mean = {mean_val:.2f}, Std = {std_val:.2f}, N = {count_val}"
    plt.plot([], [], ' ', label=stats_label)
    plt.title(title)
    plt.legend(loc="best") 
    plt.savefig(os.path.join(SAVE_DIR_EVENTS, f"{title.replace(' ', '_')}.png"), dpi=300)
    plt.close()

def main():
    global FILE_PATH, file, tree, n_entries,run_number, RUN_DIR, SAVE_DIR_EVENTS, SAVE_DIR_FULL, SAVE_DIR_CHANNEL, HITS_CSV
    run_info = get_runs("run_list.json")
    for file_path, run_number in run_info:
        # ==============================
        # LOAD ROOT FILE AND TREE
        # ==============================
        FILE_PATH = file_path
        file = ROOT.TFile.Open(FILE_PATH)
        tree = file.Get("EventTree")
        n_entries = tree.GetEntries()
        # ==============================
        # Save Directories
        # ==============================
        RUN_DIR = f"run_{run_number}"
        SAVE_DIR_EVENTS = RUN_DIR+"/event_adc_plots"
        SAVE_DIR_FULL = RUN_DIR+"/full_displays"
        HITS_CSV = os.path.join(SAVE_DIR_EVENTS, run_number+"_"+"HG_event_hits.csv")
        os.makedirs(SAVE_DIR_EVENTS, exist_ok=True)
        os.makedirs(SAVE_DIR_FULL, exist_ok=True)
        start_time = time.time()
        print(f"[INFO] Starting high gain analysis for run {run_number} with threshold {THRESHOLD}, events = {n_entries} remap={True}.")
        if os.path.isfile(HITS_CSV):
            check_run()
        else:
            analyze_run(events=n_entries, max_hits=2, remap=True, board1=BOARD1, board2=BOARD2, board3=BOARD3, threshold=THRESHOLD, veto_threshold=13.6)
        plot_run(HITS_CSV, title= f"Run {run_number} HG Hit Map")
        plot_mean_adc_histogram(HITS_CSV, title= f"Run {run_number} Mean HG ADC Histogram")
        veto_profile2d(range(n_entries), tree, veto_board=BOARD3, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative HG Veto Amplitude (mean)")
        veto_profile2d(range(n_entries), tree, veto_board=MCP1, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative HG MCP1 Amplitude (mean)")
        veto_profile2d(range(n_entries), tree, veto_board=MCP2, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative HG MCP2 Amplitude (mean)")
        end_time = time.time()
        print(f"[INFO] Analysis complete for run {run_number}.")
        print(f"[INFO] Run time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
