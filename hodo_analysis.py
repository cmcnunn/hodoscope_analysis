import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# GLOBAL CONFIGURATION
# ==============================
FILE_PATH = "/home/colin_regex/CaloXData/run1326_250820010716_TimingDAQ.root"
BOARD1 = "FERS_Board0_energyHG" #X-axis 
BOARD2 = "FERS_Board1_energyHG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6_amp" #VETO Measuments 
THRESHOLD = 4500  # Minimum ADC value for event selection

# Directories for saving plots
SAVE_DIR_EVENTS = "event_adc_plots"
SAVE_DIR_FULL = "full_displays"
SAVE_DIR_CHANNEL = "channel_adc_dis"
os.makedirs(SAVE_DIR_EVENTS, exist_ok=True)
os.makedirs(SAVE_DIR_FULL, exist_ok=True)
os.makedirs(SAVE_DIR_CHANNEL, exist_ok=True)


# ==============================
# LOAD ROOT FILE AND TREE
# ==============================
file = ROOT.TFile.Open(FILE_PATH)
tree = file.Get("EventTree")
n_entries = tree.GetEntries()

# Track low-energy events
bad_events = []

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
s_fers_vert_mapping = [
    56,48,40,32,24,16, 8, 0,
     4,12,20,28,36,44,52,60,
    58,50,42,34,26,18,10, 2,
     6,14,22,30,38,46,54,62,
    57,49,41,33,25,17, 9, 1,
     5,13,21,29,37,45,53,61,
    59,51,43,35,27,19,11, 3,
     7,15,23,31,39,47,55,63
]
######CHANGE HERE######
mapping1 = s_fers_mapping
mapping2 = s_fers_vert_mapping
#######################
def do_map(data, apply_mapping=mapping):
    """Apply channel remapping to a list of 64 ADC values."""
    if len(data) != len(smapping):
        raise ValueError("Data length and mapping length must match.")
    remapped_data = [data[i] for i in smapping]
    return remapped_data
# ==============================
# PLOTTING FUNCTIONS
# ==============================

def plot_event_adc_hist(event_id, board1=BOARD1, board2=BOARD2, save_fig=False, remap=False):
    """
    Plot ADC values for both boards for a single event as overlapping bar charts.
    """
    if event_id < 0 or event_id >= n_entries:
        raise ValueError("Event ID out of range")

    tree.GetEntry(event_id)
    energies_1 = list(getattr(tree, board1))
    energies_2 = list(getattr(tree, board2))

    # Skip low-energy events
    if max(energies_1) < THRESHOLD or max(energies_2) < THRESHOLD:
        print(f"Event {event_id} skipped (low energy)")
        bad_events.append(event_id)
        return

    if remap:
        energies_1 = do_map(energies_1,mapping1)
        energies_2 = do_map(energies_2,mapping2)
        event_label = f"{event_id}_remapped"
    else:
        event_label = str(event_id)

    print(f"Plotting Event {event_label} (Max energies: {max(energies_1)}, {max(energies_2)})")

    plt.figure(figsize=(8, 6))
    channels = range(64)
    plt.bar(channels, energies_1, color='royalblue', alpha=0.5, label=board1)
    plt.bar(channels, energies_2, color='orange', alpha=0.5, label=board2)
    plt.xlabel("Channel Number")
    plt.ylabel("ADC Count")
    plt.title(f"Event {event_label}")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_fig:
        filename = f"{SAVE_DIR_EVENTS}/event_{event_label}.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_event_adc_2dhist(event_id, board1=BOARD1, board2=BOARD2, save_fig=False, remap=False):
    """
    Plot a 2D ADC histogram for one event (board1 vs board2).
    """
    if event_id < 0 or event_id >= n_entries:
        raise ValueError("Event ID out of range")

    tree.GetEntry(event_id)
    energies_1 = list(getattr(tree, board1))
    energies_2 = list(getattr(tree, board2))

    if max(energies_1) < THRESHOLD or max(energies_2) < THRESHOLD:
        print(f"Event {event_id} skipped (low energy)")
        return

    if remap:
        energies_1 = do_map(energies_1,mapping1)
        energies_2 = do_map(energies_2,mapping2)
        event_label = f"{event_id}_remapped"
    else:
        event_label = str(event_id)

    print(f"Plotting 2D ADC histogram for Event {event_label}")

    Z = (np.array(energies_1)[:, None] + np.array(energies_2)[None, :]) / 2

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(label='Mean ADC Count')
    plt.xlabel(f"{board1} Channel Number")
    plt.ylabel(f"{board2} Channel Number")
    plt.title(f"2D ADC Histogram Event {event_label}")

    if save_fig:
        filename = f"{SAVE_DIR_EVENTS}/event_{event_label}_2dhist.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")
    else:
        plt.show()
    plt.close()


def hodoscopeHeatmap(event_list = None, board1=BOARD1, board2=BOARD2, save_fig=False, remap=False, title = "2D_ADC_Histogram"):
    print("Running hodoscopeHeatmap...")
    """
    Generate a 64x64 heatmap for all events (averaged ADC values).
    """
    Z = np.zeros((64, 64))
    valid_events = 0
    filename = "full_run_adc_heatmap"

    if event_list == None:
        event_list = range(n_entries)

    for event_id in event_list:
        tree.GetEntry(event_id)
        energies_1 = list(getattr(tree, board1))
        energies_2 = list(getattr(tree, board2))

        if max(energies_1) < THRESHOLD or max(energies_2) < THRESHOLD:
            bad_events.append(event_id)
            continue

        if remap:
            energies_1 = do_map(energies_1,mapping1)
            energies_2 = do_map(energies_2,mapping2)

        Z += (np.array(energies_2)[:, None] + np.array(energies_1)[None, :])
        valid_events += 1

    if valid_events > 0:
        Z /= valid_events

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(label='Average ADC Count')
    plt.xlabel(f"{board1} Channel Number")
    plt.ylabel(f"{board2} Channel Number")
    plt.title(f"{title}")

    if remap: 
        filename += "_remapped"

    if save_fig:
        filename = f"{SAVE_DIR_FULL}/{filename}.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")
    else:
        plt.show()
    plt.close()
    mean_val = np.mean(Z) if valid_events > 0 else 0
    print(f"[INFO] Plotted {valid_events} events. Mean ADC: {mean_val:.2f}")
    print(f"Total bad events skipped: {len(bad_events)}")


def hodoscopeHitmap(event_list=None, board1=BOARD1, board2=BOARD2, save_fig=False, remap=False, title ='2D_Coincidence_Hitmap'):
    """
    Generate a 64x64 coincidence map based on channels above threshold.

    Args:
        event_list (list[int] or None): List of event IDs to process. If None, process all events.
        board1 (str): Branch name for the first board.
        board2 (str): Branch name for the second board.
        save_fig (bool): Save the figure to file if True.
        remap (bool): Apply channel remapping if True.
    """
    print("Running hodoscopeHitmap...")
    bad = []

    if remap:
        title += "_remapped"
    
    if event_list is not None and len(event_list) < n_entries:
        title += "_subset"

    Z = np.zeros((64, 64))
    filename = title

    # Determine which events to process
    if event_list is None:
        event_list = range(n_entries)

    for event_id in event_list:
        tree.GetEntry(event_id)
        energies_1 = list(getattr(tree, board1))
        energies_2 = list(getattr(tree, board2))

        # Skip low-energy events
        if max(energies_1) < THRESHOLD or max(energies_2) < THRESHOLD:
            bad.append(event_id)
            continue

        if remap:
            energies_1 = do_map(energies_1,mapping1)
            energies_2 = do_map(energies_2,mapping2)

        # Find channels above threshold
        good_x = [i for i, val in enumerate(energies_1) if val >= THRESHOLD]
        good_y = [j for j, val in enumerate(energies_2) if val >= THRESHOLD]

        # Increment coincidence matrix
        for x in good_x:
            for y in good_y:
                Z[y, x] += 1
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(Z, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Coincidence Count')
    plt.title(f"{title}")
    plt.xlabel(f'{board1} Channel Number')
    plt.ylabel(f'{board2} Channel Number')
    plt.tight_layout()
    mean_val = np.mean(Z) if np.sum(Z) > 0 else 0
    print(f"[INFO] Plotted {len(event_list)-len(bad_events)} events. Mean coincidences per cell: {mean_val:.2f}")
    if save_fig:
        filename = f"{SAVE_DIR_FULL}/{filename}.png"
        plt.savefig(filename, dpi=300)
        print(f"Hitmap saved: {filename}")
    else:
        plt.show()
    plt.close()

def plot_channel_adc_distribution(events = n_entries, board1=BOARD1, board2=BOARD2, save_fig=False, remap=False):
    print("Running plot_channel_adc_distribution...")
    """
    Plot ADC value distribution for each channel across all events.
    """
    all_energies_1 = [[] for _ in range(64)]
    all_energies_2 = [[] for _ in range(64)]

    for event_id in events:
        tree.GetEntry(event_id)
        energies_1 = list(getattr(tree, board1))
        energies_2 = list(getattr(tree, board2))

        if max(energies_1) < THRESHOLD or max(energies_2) < THRESHOLD:
            continue

        if remap:
            energies_1 = do_map(energies_1,mapping1)
            energies_2 = do_map(energies_2,mapping2)

        for i in range(64):
            all_energies_1[i].append(energies_1[i])
            all_energies_2[i].append(energies_2[i])

    for i in range(64):
        total_count_1 = len(all_energies_1[i])
        total_count_2 = len(all_energies_2[i])

        plt.figure(figsize=(12, 8))
        plt.hist(all_energies_1[i], bins=32, alpha=0.5, label=f'{board1} (count={total_count_1})')
        plt.hist(all_energies_2[i], bins=32, alpha=0.5, label=f'{board2} (count={total_count_2})')
        plt.yscale('log')
        plt.xlabel('ADC Count')
        plt.ylabel('Frequency')
        plt.title(f'ADC Distribution for Channel {i}\n{board1} vs {board2}')
        plt.legend()
        plt.tight_layout()

        if save_fig:
            filename = f"{SAVE_DIR_CHANNEL}/channel_adc_distribution_ch{i}.png"
            plt.savefig(filename, dpi=300)
            print(f"Figure saved: {filename}")
        else:
            plt.show()
        plt.close()
    print("ADC distribution plots generated.")

# ==============================
#  SECTION: PEAK ANALYSIS
# ==============================
import csv

HITS_CSV = os.path.join(SAVE_DIR_EVENTS, "event_hits.csv")

def detect_hits_per_event(event=None, eventrange=None, board1="FERS_Board0_energyHG", board2="FERS_Board1_energyHG",
                          threshold=THRESHOLD, remap=False, save_csv=True):
    global mapping, n_entries
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
        print(f"[INFO] Processed event {event} for hit analysis")
    elif eventrange is not None:
        for event_id in range(eventrange):
            process_event(event_id)
            if event_id % 100 == 0:
                print(f"[INFO] Processed {event_id}/{n_entries} events for hit analysis")
    else:
        raise ValueError("Either 'event' or 'eventrange' must be provided.")

    if save_csv:
        with open(HITS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "num_hits", "channels_with_hits", "mean_adc"])
            writer.writerows(results)
        print(f"[INFO] Peak detection results saved to {HITS_CSV}")

    mean_val = np.mean(multiplicities)
    # Plot multiplicity histogram
    plt.figure(figsize=(8, 5))
    plt.hist(multiplicities, bins=range(0, max(multiplicities)+2), alpha=0.7)
    plt.xlabel("Number of Hits (Channels Above Threshold)")
    plt.ylabel("Number of Events")
    plt.title(f"Peak Multiplicity Distribution (Threshold = {threshold})")
    plt.text(0.95, 0.95, f"Mean: {mean_val:.2f}",
         transform=plt.gca().transAxes,
         ha='right', va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_EVENTS, "peak_multiplicity_histogram.png"))
    plt.close()
    print(f"[INFO] Peak multiplicity histogram saved as {SAVE_DIR_EVENTS}/peak_multiplicity_histogram.png.")
# ==============================
# FIND GOOD EVENTS
# ==============================
def find_good_events(csv_file, max_hits=2):
    good_events = []
    adc_values = []
    with open(csv_file, 'r', newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_hits = int(row['num_hits'])
            if num_hits <= max_hits:
                good_events.append(int(row['event_id']))
                adc_values.append(float(row['mean_adc']))

    print(f"[INFO] Found {len(good_events)} good events with ≤ {max_hits} hits.")
    if adc_values:
        mean_adc = sum(adc_values) / len(adc_values)
        print(f"[INFO] Mean ADC of good events: {mean_adc:.2f}")
    else:
        print("[INFO] No good events found, mean ADC not available.")
    return good_events

# ==============================
# Compare VETO 
# ==============================

def getUpstreamVeto(events, veto_board = BOARD3, veto_threshold = 600):
    vetoed = []
    good = []
    for event_id in range(events):
        tree.GetEntry(event_id)
        veto_energy = getattr(tree,veto_board)
        if veto_energy < veto_threshold:
            good.append(event_id)
        else:
            vetoed.append(event_id)

    print("[INFO] Done Vetoing Events")
    print(f"[INFO] Number of beam hits {len(good)}")

    return good, vetoed

# ==============================
# MAIN EXECUTION
# ==============================
import time

def main():
    
    print("[INFO] Starting Hodoscope Data Analysis...")
    start_time = time.time()

    # Step 1: Apply veto selection
    print("[INFO] Step 1: Applying upstream veto...")
    good, vetoed = getUpstreamVeto(n_entries)
    print(f"[INFO] Veto complete. Good events: {len(good)}, Vetoed events: {len(vetoed)}")

    # Step 2: Find good events by hits
    print("[INFO] Step 2: Apply peak check")
    detect_hits_per_event(eventrange=n_entries, remap=True, save_csv=True)
    good_hits = find_good_events(HITS_CSV, max_hits=2)

    # Step 3: Get coincidence between veto and peak selected events
    print("[INFO] Step 3: Finding coincidence between veto and hit selected events...")
    good_good = list(set(good) & set(good_hits))
    print(f"[INFO] Events passing both veto and hit criteria: {len(good)}")

    # Step 4: Generate coincidence hitmaps
    
    print("[INFO] Step 4: Generating coincidence hitmaps...")
    hodoscopeHitmap(good, save_fig=True, remap=True, title='2D_coincidence_hitmap_veto')
    hodoscopeHitmap(good_hits, save_fig=True, remap=True, title='2D_coincidence_hitmap_peakdetection')
    hodoscopeHitmap(good_good, save_fig=True, remap=True, title='2D_coincidence_hitmap_bothcuts')
    print("[INFO] Coincidence hitmap complete.")
     # Step 5: Generate channel distributions
    print("[INFO] Step 5: Generating channel adc distributions")
    plot_channel_adc_distribution(events=good_good, save_fig=True, remap=True)
     # Step 6: Generate coincidence hitmaps
    print("[INFO] Step 6: Plot evnt ADC 1D and 2D Histograms")
    for event_id in good_good: #Plot individual events
        plot_event_adc_hist(event_id, save_fig=True, remap=True)
        plot_event_adc_2dhist(event_id, save_fig=True, remap=True)

    elapsed_time = time.time() - start_time
    print(f"[INFO] Analysis complete in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()






