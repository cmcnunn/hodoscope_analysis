import ROOT
import matplotlib.pyplot as plt
import json

# ==============================
# GLOBAL CONFIGURATION
# ==============================
BOARD1 = "FERS_Board0_energyHG" #X-axis 
BOARD2 = "FERS_Board1_energyHG" #Y-axis
BOARD3 = "DRS_Board7_Group1_Channel6" #VETO Measuments 
MCP1 = "DRS_Board0_Group3_Channel6" #MCP1 Measuments
MCP2 = "DRS_Board0_Group3_Channel7" #MCP2 Measuments
THRESHOLD = 5500  # Minimum ADC value for event selection

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

