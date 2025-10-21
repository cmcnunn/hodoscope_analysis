import ROOT
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import csv
import json
import time
from utils.run_utils import get_runs
from utils.analysis_utils import analyze_run, check_run, plot_run, veto_profile2d, plot_mean_adc_histogram, do_map

# ==============================
# GLOBAL CONFIGURATION
# ==============================
BOARD1 = "FERS_Board0_energyLG" #X-axis 
BOARD2 = "FERS_Board1_energyLG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6" #VETO Measuments 
MCP1 = "DRS_Board0_Group3_Channel6" #MCP1 Measuments
MCP2 = "DRS_Board0_Group3_Channel7" #MCP2 Measuments
THRESHOLD = 200  # Minimum ADC value for LOW GAIN event selection

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
        HITS_CSV = os.path.join(SAVE_DIR_EVENTS, run_number+"_"+"LG_event_hits.csv")
        os.makedirs(SAVE_DIR_EVENTS, exist_ok=True)
        os.makedirs(SAVE_DIR_FULL, exist_ok=True)
        start_time = time.time()
        print(f"[INFO] Starting analysis for run {run_number} with threshold {THRESHOLD}, events = {n_entries} remap={True}.")
        if os.path.isfile(HITS_CSV):
            check_run()
        else:
            analyze_run(events=n_entries, max_hits=2, remap=True, board1=BOARD1, board2=BOARD2, board3=BOARD3, threshold=THRESHOLD, veto_threshold=13.6)
        plot_run(HITS_CSV, title= f"Run {run_number} LG Hit Map")
        plot_mean_adc_histogram(HITS_CSV, title= f"Run {run_number} Mean LG ADC Histogram")
        veto_profile2d(range(n_entries), tree, veto_board=BOARD3, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative LG Veto Amplitude (mean)")
        veto_profile2d(range(n_entries), tree, veto_board=MCP1, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative LG MCP1 Amplitude (mean)")
        veto_profile2d(range(n_entries), tree, veto_board=MCP2, bins=(64, 64), xr=(0, 64), yr=(0, 64), min_count=2, title=f"Run {run_number} Cumulative LG MCP2 Amplitude (mean)")
        end_time = time.time()
        print(f"[INFO] Analysis complete for run {run_number}.")
        print(f"[INFO] Run time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()