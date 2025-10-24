import ROOT
import os 
import json 
import numpy as np
import matplotlib.pyplot as plt
import time 

HG_FERS0 = "FERS_Board0_energyHG" #X-axis 
HG_FERS1 = "FERS_Board1_energyHG" #Y-axis
LG_FERS0 = "FERS_Board0_energyLG" #X-axis
LG_FERS1 = "FERS_Board1_energyLG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6" #VETO Measuments 
MCP1 = "DRS_Board0_Group3_Channel6" #MCP1 Measuments
MCP2 = "DRS_Board0_Group3_Channel7" #MCP2 Measuments
HG_THRESHOLD = 5500  # Minimum ADC value for HIGH GAIN event selection
LG_THRESHOLD = 225   # Minimum ADC value for LOW GAIN event selection
#SNAKE MAPPING + FERS MAPPING 
fers0_mapping = [
     0, 8,16,24,32,40,48,56,
    60,52,44,36,28,20,12, 4,
     2,10,18,26,34,42,50,58,
    62,54,46,38,30,22,14, 6,
     1, 9,17,25,33,41,49,57,
    61,53,45,37,29,21,13, 5,
     3,11,19,27,35,43,51,59,
    63,55,47,39,31,23,15, 7
]


fers1_mapping = [
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
mapping1 = fers0_mapping
mapping2 = fers1_mapping
#######################

def do_map(data, apply_mapping):
    """Apply channel remapping to a list of 64 ADC values."""
    if len(data) != len(apply_mapping):
        raise ValueError("Data length and mapping length must match.")
    remapped_data = [data[i] for i in apply_mapping]
    return remapped_data

def getupstreamVeto(entry,tree, veto_threshold=13.6):
    global BOARD3
    """Check the veto for one event"""
    tree.GetEntry(entry)
    veto_energy = list(getattr(tree,BOARD3))
    veto_energy = np.min(veto_energy)
    return veto_energy < veto_threshold

def process_event(entry,tree):
    global THRESHOLD, HG_FERS0, HG_FERS1
    tree.GetEntry(entry)
    energies_1 = list(getattr(tree, HG_FERS0))
    if max(energies_1) > 8000:
        energies_1 = list(getattr(tree, LG_FERS0))
    energies_2 = list(getattr(tree, HG_FERS1))
    if max(energies_2) > 8000:
        energies_2 = list(getattr(tree, LG_FERS1))
    energies_1 = do_map(energies_1,mapping1)
    energies_2 = do_map(energies_2,mapping2)
    return energies_1, energies_2

def detect_hits(x_adc, y_adc):
    global HG_THRESHOLD, LG_THRESHOLD
    if max(x_adc) > HG_THRESHOLD:
        threshold_x = HG_THRESHOLD
    else:
        threshold_x = LG_THRESHOLD
    if max(y_adc) > HG_THRESHOLD:
        threshold_y = HG_THRESHOLD
    else:
        threshold_y = LG_THRESHOLD
    hit_x = [i for i, adc in enumerate(x_adc) if adc >= threshold_x]
    hit_y = [i for i, adc in enumerate(y_adc) if adc >= threshold_y]
    multiplicity_x = len(hit_x)
    multiplicity_y = len(hit_y)
    if multiplicity_x > 2 or multiplicity_y > 2:
        hit_x = []
        hit_y = []
    return hit_x, hit_y

def multirun_analysis(run_list, title = "MultiRunAnalysis"):
    # Load run file list 
    OUTPUT_DIR = title+"_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open("run_list.json", "r") as f:
        run_files = json.load(f)
    # Loop over runs
    Z = np.zeros((64, 64))

    for run in run_list:
        if run not in run_files:
            print(f"Run {run} not found in run list.")
            continue
        file_path = run_files[run]["file"]
        if not os.path.exists(file_path):
            print(f"File for run {run} does not exist: {file_path}")
            continue
        print(f"Processing run {run} from file {file_path}")
        # Open ROOT file
        file = ROOT.TFile.Open(file_path)
        tree = file.Get("EventTree")
        n_entries = tree.GetEntries()
        if not file or file.IsZombie():
            print(f"Failed to open ROOT file for run {run}: {file_path}")
            continue
        for i in range(n_entries):
            x_adc, y_adc,= process_event(i, tree)
            hit_x, hit_y = detect_hits(x_adc, y_adc)
            for x in hit_x:
                for y in hit_y:
                    Z[x, y] += 1

                    
        file.Close()

    return Z, OUTPUT_DIR

def create_2Dhist(Z, title, OUTPUT_DIR):
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin="lower", cmap="viridis")
    plt.colorbar(label="Counts")
    plt.xlabel("Board 1 Channels")
    plt.ylabel("Board 2 Channels")
    plt.title(title)
    plt.savefig(f"{OUTPUT_DIR}+{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

def main():
    start = time.time()
    run_list = ["1510", "1526", "1527"]  # Example run numbers
    title = "MultiRunAnalysis_1510_1526_1527"
    Z, OUTPUT_DIR = multirun_analysis(run_list, title)
    create_2Dhist(Z, title, OUTPUT_DIR)
    end = time.time()
    full_time = end - start
    print(f"Analysis completed in {full_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
