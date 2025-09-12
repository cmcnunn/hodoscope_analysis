# hodoscope_analysis
This is the repository of Hodoscope analysis and plotting code for HG-Dream 
## Features

- Plot ADC distributions per Channel
- Plot event ADC 1D and 2D histograms
- Plot Heatmaps and Hitmaps for ranges of events
- Detect Hits per event and creat csv that shows hit channels + mean adc per event
- Analyse using upstream VETO 
## Requirements

- Python 3.7+
- ROOT
- `matplotlib`
- `numpy`
- `os`
## Function Descriptions

### do_map(data, apply_mapping=mapping)
Applies a channel remapping to a list of 64 ADC values based on the specified mapping scheme.

---

### getUpstreamVeto(events, veto_board, veto_threshold)
Applies an upstream veto cut to events.  
- Returns lists of good and vetoed events.  
- Reports number of surviving beam hits.

---

### detect_hits_per_event(event=None, eventrange=None, board1, board2, threshold, remap=False, save_csv=True)
Detects hits (channels above threshold) for events and saves results.  
- Records multiplicity, hit channels, and mean ADC per event.  
- Saves results to `event_hits.csv`.  
- Produces a hit multiplicity histogram.

---

### find_good_events(csv_file, max_hits=2)
Reads hit detection results and selects “good” events with ≤ `max_hits`.  
- Returns a list of event IDs.  
- Reports mean ADC of good events.

---

### analyze_event(event_id, board1=BOARD1, board2=BOARD2, threshold=THRESHOLD, remap=False)
Reads event ADC, remaps (if needed) and records hits if over a threshold

---

### plot_event_2Dhist(hit_x, hit_y, save_fig=False, event_id=None)
Takes in output from analyze_event and plots a single event

---

### plot_allevents_2Dhist(events = n_entries, save_fig=False, plot_single_events=False)
Compiles analyze_event to plot full run hit data. Optionally plots single events. 

### main()
Main pipeline for hodoscope analysis. Runs:  
1. Hit detection and filtering  
2. Upstream veto selection 
4. Coincidence hitmap generation  
