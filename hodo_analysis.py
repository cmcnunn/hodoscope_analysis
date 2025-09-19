import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time

# ==============================
# GLOBAL CONFIGURATION
# ==============================
FILE_PATH = "/home/colin_regex/CaloXData/run1326_250820010716_TimingDAQ.root"
BOARD1 = "FERS_Board0_energyHG" #X-axis 
BOARD2 = "FERS_Board1_energyHG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6_amp" #VETO Measuments 
# [100,1000,1500,4000,8000]
THRESHOLD = 5500  # Minimum ADC value for event selection
# Directories for saving plots
SAVE_DIR_EVENTS = "event_adc_plots"
SAVE_DIR_FULL = "full_displays"
SAVE_DIR_CHANNEL = "channel_adc_dis"
HITS_CSV = os.path.join(SAVE_DIR_EVENTS, "event_hits.csv")
os.makedirs(SAVE_DIR_EVENTS, exist_ok=True)
os.makedirs(SAVE_DIR_FULL, exist_ok=True)
os.makedirs(SAVE_DIR_CHANNEL, exist_ok=True)


# ==============================
# LOAD ROOT FILE AND TREE
# ==============================
file = ROOT.TFile.Open(FILE_PATH)
tree = file.Get("EventTree")
n_entries = tree.GetEntries()

# ==============================
# CHANNEL MAPPING CONFIGURATIONS
# ==============================

#Recommended Mapping::
#BOARD1 = SNAKE MAPPING + FERS MAPPING
#BOARD2 = SNAKE MAPPING + FERS MAPPING VERTICAL FLIP

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

#SNAKE + FERS + VERT FLIP
s_fers_flip_mapping = [
     7,15,23,31,39,47,55,63,
    59,51,43,35,27,19,11, 3,
     5,13,21,29,37,45,53,61,
    57,49,41,33,25,17, 9, 1,
     6,14,22,30,38,46,54,62,
    58,50,42,34,26,18,10, 2,
     4,12,20,28,36,44,52,60,
    56,48,40,32,24,16, 8, 0
]

######CHANGE HERE######
mapping1 = s_fers_mapping
mapping2 = s_fers_flip_mapping
#######################


# ==============================
# CHANNEL REMAPPING FUNCTION
# ==============================

def do_map(data, apply_mapping=s_fers_mapping):
    """Apply channel remapping to a list of 64 ADC values."""
    if len(data) != len(apply_mapping):
        raise ValueError("Data length and mapping length must match.")
    remapped_data = [data[i] for i in apply_mapping]
    return remapped_data

# ==============================
# EVENT ANALYSIS FUNCTIONS
# ==============================

def getUpstreamVeto(events, veto_board = BOARD3, veto_threshold = 600):
    vetoed = []
    good = []
    for event_id in events:
        tree.GetEntry(event_id)
        veto_energy = getattr(tree,veto_board)
        if veto_energy < veto_threshold:
            good.append(event_id)
        else:
            vetoed.append(event_id)

    print("[INFO] Done Vetoing Events")
    print(f"[INFO] Number of beam hits {len(good)}; Veto threshold {veto_threshold}")

    return good, vetoed

def plotVetoDistribution(events, veto_board = BOARD3, bins=100, range=(0, 3000), save_fig=False):
    veto_energies = []
    for event_id in events:
        tree.GetEntry(event_id)
        veto_energy = getattr(tree,veto_board)
        veto_energies.append(veto_energy)

    plt.figure(figsize=(8, 5))
    plt.hist(veto_energies, bins=bins, range=range, alpha=0.7, color='blue')
    plt.xlabel("Veto Energy (ADC)")
    plt.ylabel("Number of Events")
    plt.title("Upstream Veto Energy Distribution")
    if save_fig:
        plt.savefig(os.path.join(SAVE_DIR_FULL, "veto_energy_distribution.png"))
        print(f"[INFO] Veto energy distribution saved as {SAVE_DIR_FULL}/veto_energy_distribution.png.")
    else:
        plt.show()
    plt.close()

def getVetomedian(events, veto_board = BOARD3):
    veto_energies = []
    for event_id in events:
        tree.GetEntry(event_id)
        veto_energy = getattr(tree,veto_board)
        veto_energies.append(veto_energy)

    median_veto = np.median(veto_energies)
    print(f"[INFO] Veto Mean: {median_veto:.2f}")
    return median_veto
#!! Make sure to only run detect_hits_per_event once to avoid overwriting the CSV file !!

def detect_hits_per_event(event=None, eventrange=None, board1="FERS_Board0_energyHG", board2="FERS_Board1_energyHG",
                          threshold=THRESHOLD, remap=False, save_csv=True):
    results = []
    multiplicities = []

    def process_event(event_id):
        tree.GetEntry(event_id)
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

        results.append([event_id, multiplicity, hits, mean_adc])
        multiplicities.append(multiplicity)

    if event is not None:
        process_event(event)
    elif eventrange is not None:
        for event_id in range(eventrange):
            process_event(event_id)
    else:
        raise ValueError("Either 'event' or 'eventrange' must be provided.")

    if save_csv:
        with open(HITS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "num_hits", "channels_with_hits", "mean_adc"])
            writer.writerows(results)
        print(f"[INFO] Hit detection results saved to {HITS_CSV}")

    mean_val = np.mean(multiplicities)
    # Plot multiplicity histogram
    plt.figure(figsize=(8, 5))
    plt.hist(multiplicities, bins=range(0, max(multiplicities)+2), alpha=0.7)
    plt.xlabel("Number of Hits (Channels Above Threshold)")
    plt.ylabel("Number of Events")
    plt.title(f"Hit Multiplicity Distribution (Threshold = {threshold})")
    plt.text(0.95, 0.95, f"Mean: {mean_val:.2f}",
         transform=plt.gca().transAxes,
         ha='right', va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_EVENTS, "hit_multiplicity_histogram.png"))
    plt.close()
    print(f"[INFO] Hit multiplicity histogram saved as {SAVE_DIR_EVENTS}/hit_multiplicity_histogram.png.")

def find_good_hit_events(csv_file, max_hits=2):
    good_hit_events = []
    adc_values = []
    with open(csv_file, 'r', newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_hits = int(row['num_hits'])
            if num_hits <= max_hits and num_hits > 0:
                good_hit_events.append(int(row['event_id']))
                adc_values.append(float(row['mean_adc']))

    print(f"[INFO] Found {len(good_hit_events)} good events with ≤ {max_hits} hits.")
    if adc_values:
        mean_adc = sum(adc_values) / len(adc_values)
        print(f"[INFO] Mean ADC of good events: {mean_adc:.2f}")
    else:
        print("[INFO] No good events found, mean ADC not available.")
    return good_hit_events

def analyze_event(event_id, board1=BOARD1, board2=BOARD2, threshold=THRESHOLD, remap=False):
    Z = np.zeros((64, 64))  # Global variable to hold the 2D histogram data
    tree.GetEntry(event_id)
    energies_1 = list(getattr(tree, board1))
    energies_2 = list(getattr(tree, board2))

    if remap:
        energies_1 = do_map(energies_1,mapping1)
        energies_2 = do_map(energies_2,mapping2)

    hits_x = [i for i, val in enumerate(energies_1) if val > threshold]
    hits_y = [i for i, val in enumerate(energies_2) if val > threshold]

    for x in hits_x:
        for y in hits_y:
            Z[y, x] += 1  # Note: y corresponds to rows, x to columns
    return hits_x, hits_y

# ==============================
# EVENT Plotting FUNCTIONS
# ==============================

def plot_event_2Dhist(hit_x, hit_y, save_fig=False, event_id=None):
    plt.figure(figsize=(8, 6))
    plt.hist2d(hit_x, hit_y, bins=[64, 64], range=[[0, 64], [0, 64]], cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('Board 1 Channels (X-axis)')
    plt.ylabel('Board 2 Channels (Y-axis)')
    plt.title('2D Histogram of Hit Channels')
    if save_fig and event_id is not None:
        plt.savefig(os.path.join(SAVE_DIR_EVENTS, f"event_{event_id}_hits.png"))
        print(f"[INFO] Event {event_id} hit histogram saved as {SAVE_DIR_EVENTS}/event_{event_id}_hits.png.")
    else: 
        plt.show()
    plt.close()

def plot_allevents_2Dhist(events = n_entries, save_fig=False, plot_single_events=False, title="Cumulative 2D Histogram of All Events"):
    Z = np.zeros((64, 64))  # Global variable to hold the 2D histogram data
    for i in events:
        hits_x, hits_y = analyze_event(i, remap=True)
        for x in hits_x:
            for y in hits_y:
                Z[y, x] += 1  # Note: y corresponds to rows, x to columns
        if plot_single_events:
            plot_event_2Dhist(hits_x, hits_y, save_fig=save_fig, event_id=i)
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin='lower', cmap='viridis', extent=[0, 64, 0, 64])
    plt.colorbar(label='Counts')
    plt.xlabel('Board 1 Channels (X-axis)')
    plt.ylabel('Board 2 Channels (Y-axis)')
    plt.title(title)
    if save_fig:
        filename = title.replace(" ", "_").lower()
        plt.savefig(os.path.join(SAVE_DIR_FULL, filename + ".png"))
        print(f"[INFO] Cumulative 2D histogram saved as {SAVE_DIR_FULL}/{filename}.png.")
    else:
        plt.show()
    plt.close()

def get_efficiency_plot(hit_range, save_fig=False, veto_thresh=600):
    n_hits = []
    efficiencies = []
    vetted_events, _ = getUpstreamVeto(range(n_entries), veto_threshold=veto_thresh)

    for max_hits in range(hit_range):
        good_events = find_good_hit_events(HITS_CSV, max_hits=max_hits)
        vetted_good_events, _ = getUpstreamVeto(good_events, veto_threshold=veto_thresh)
        efficiency = 100 * len(vetted_good_events) / len(vetted_events) if vetted_events else 0
        n_hits.append(max_hits)
        efficiencies.append(efficiency)
        print(f"[INFO] Efficiency for max {max_hits} hits: {efficiency:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(n_hits, efficiencies, marker='o')
    plt.xlabel("Maximum Number of Hits Allowed")
    plt.ylabel("Efficiency (%)")
    plt.title(f"Efficiency vs. Maximum Number of Hits Threshold = {THRESHOLD}, Veto = {veto_thresh:.0f}")
    plt.grid(True)
    if save_fig:
        plt.savefig(os.path.join(SAVE_DIR_FULL, "efficiency_vs_max_hits.png"))
        print(f"[INFO] Efficiency plot saved as {SAVE_DIR_FULL}/efficiency_vs_max_hits.png.")
    else:
        plt.show()
    plt.close()

def get_efficiency_per_threshold(thresholds, veto_thresh=600, hits_max=2):
    efficiencies = []
    vetted_events, _ = getUpstreamVeto(range(n_entries), veto_threshold=veto_thresh)

    for thresh in thresholds:
        detect_hits_per_event(eventrange=n_entries, threshold=thresh, remap=True, save_csv=True)
        good_events = find_good_hit_events(HITS_CSV, max_hits=hits_max)
        vetted_good_events, _ = getUpstreamVeto(good_events, veto_threshold=veto_thresh)
        efficiency = 100 * len(vetted_good_events) / len(vetted_events) if vetted_events else 0
        efficiencies.append(efficiency)
        print(f"[INFO] Efficiency for threshold {thresh}: {efficiency:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, efficiencies, marker='o')
    plt.xlabel("ADC Threshold")
    plt.ylabel("Efficiency (%)")
    plt.title(f"Efficiency vs. ADC Threshold, Veto = {veto_thresh:.0f}")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR_FULL, "efficiency_vs_adc_threshold.png"))
    print(f"[INFO] Efficiency vs. ADC threshold plot saved as {SAVE_DIR_FULL}/efficiency_vs_adc_threshold.png.")
    plt.close()

def get_efficiency_per_veto(veto_thresholds, hits_max=2, threshold=THRESHOLD):
    efficiencies = []

    for veto_thresh in veto_thresholds:
        vetted_events, _ = getUpstreamVeto(range(n_entries), veto_threshold=veto_thresh)
        good_events = find_good_hit_events(HITS_CSV, max_hits=hits_max)
        vetted_good_events, _ = getUpstreamVeto(good_events, veto_threshold=veto_thresh)
        efficiency = 100 * len(vetted_good_events) / len(vetted_events) if vetted_events else 0
        efficiencies.append(efficiency)
        print(f"[INFO] Efficiency for veto threshold {veto_thresh}: {efficiency:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(veto_thresholds, efficiencies, marker='o')
    plt.xlabel("Veto Threshold")
    plt.ylabel("Efficiency (%)")
    plt.title(f"Efficiency vs. Veto Threshold, Hits ≤ {hits_max}, ADC Threshold = {threshold}")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR_FULL, "efficiency_vs_veto_threshold.png"))
    print(f"[INFO] Efficiency vs. Veto threshold plot saved as {SAVE_DIR_FULL}/efficiency_vs_veto_threshold.png.")
    plt.close()

# ==============================
# MAIN ANALYSIS WORKFLOW
# ==============================
def main():
    start_time = time.time()
    print("[INFO] Starting analysis...")
    # Step 1: Detect hits and save to CSV
    detect_hits_per_event(eventrange=n_entries, remap=True)
    # Step 2: Find good hit events (≤ 2 hits)
    good_events = find_good_hit_events(HITS_CSV, max_hits=2)
    # Step 3: Apply upstream veto
    vetted_good_events, vetoed_good_events = getUpstreamVeto(good_events, veto_threshold=13.5)
    vetted_events, _ = getUpstreamVeto(range(n_entries), veto_threshold=13.5)
    # Step 4: Plot cumulative 2D histogram of good events
    plot_allevents_2Dhist(events=vetted_good_events, save_fig=True, plot_single_events=False, title="Cumulative 2D Histogram of Good Events hits ≤ 2")
    plot_allevents_2Dhist(events=vetoed_good_events, save_fig=True, plot_single_events=False, title="Cumulative 2D Histogram of Vetoed Events hits ≤ 2")
    print("Efficiency after veto: {:.2f}%".format(100 * len(vetted_good_events) / len(vetted_events) if vetted_events else 0))
    # Step 5: Generate efficiency plots
    thresholds = np.linspace(1000, 8000, 15)
    veto_thresholds = np.linspace(10, 3000, 15)
    #get_efficiency_plot(hit_range=10, save_fig=True, veto_thresh=13.5)
    get_efficiency_per_threshold(thresholds, veto_thresh=13, hits_max=2)
    get_efficiency_per_veto(veto_thresholds, hits_max=2, threshold=THRESHOLD)
    print("[INFO] Analysis complete.")
    end_time = time.time()
    print(f"[INFO] Total analysis time: {end_time - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()