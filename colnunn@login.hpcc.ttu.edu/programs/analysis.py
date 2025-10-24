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
HG_BOARD1 = "FERS_Board0_energyHG" #X-axis 
HG_BOARD2 = "FERS_Board1_energyHG" #Y-axis
LG_BOARD1 = "FERS_Board0_energyHG" #X-axis 
LG_BOARD2 = "FERS_Board1_energyHG" #Y-axis
VETO = "DRS_Board7_Group1_Channel6" #VETO Measuments 
MCP1 = "DRS_Board0_Group3_Channel6" #MCP1 Measuments
MCP2 = "DRS_Board0_Group3_Channel7" #MCP2 Measuments
HG_THRESHOLD = 5500  # Minimum ADC value for HIGH GAIN event selection
LG_THRESHOLD = 225  # Minimum ADC value for LOW GAIN event selection
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
def do_map(data, apply_mapping):
    """Apply channel remapping to a list of 64 ADC values."""
    if len(data) != len(apply_mapping):
        raise ValueError("Data length and mapping length must match.")
    remapped_data = [data[i] for i in apply_mapping]
    return remapped_data

def getupstreamVeto(entry, veto_board=VETO, veto_threshold=13.6):
    """Check the veto for one event"""
    tree.GetEntry(entry)
    veto_energy = list(getattr(tree,veto_board))
    veto_energy = np.min(veto_energy)
    return veto_energy < veto_threshold

def getupstreamMCPs(entry, mcp1_board=MCP1, mcp2_board=MCP2, mcp_threshold=13.6):
    """Check the MCPs for one event"""
    tree.GetEntry(entry)
    mcp1_energy = list(getattr(tree,mcp1_board))
    mcp2_energy = list(getattr(tree,mcp2_board))
    mcp1_energy = np.min(mcp1_energy)
    mcp2_energy = np.min(mcp2_energy)
    return mcp1_energy, mcp2_energy

