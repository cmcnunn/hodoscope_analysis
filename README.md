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
- `matplotlib`
- `numpy`
- `os`
## Function Descriptions

### do_map(data, smapping=mapping)
Applies a channel remapping to a list of 64 ADC values based on the specified mapping scheme.

---

### plot_event_adc_hist(event_id, board1, board2, save_fig=False, remap=False)
Plots ADC values for two boards for a single event as overlapping bar charts.  
- Skips low-energy events.  
- Can remap channels.  
- Optionally saves plots to disk.

---

### plot_event_adc_2dhist(event_id, board1, board2, save_fig=False, remap=False)
Generates a 2D ADC histogram (board1 vs board2) for one event.  
- Uses average ADC counts.  
- Supports channel remapping and saving.

---

### hodoscopeHeatmap(event_list=None, board1, board2, save_fig=False, remap=False, title="2D_ADC_Histogram")
Creates a **64x64 averaged heatmap** of ADC counts across many events.  
- Optionally restricts to a subset of events.  
- Applies threshold filtering and channel remapping.  
- Reports mean ADC value and number of valid events.

---

### hodoscopeHitmap(event_list=None, board1, board2, save_fig=False, remap=False, title="2D_Coincidence_Hitmap")
Builds a **coincidence map** of channels above threshold between two boards.  
- Coincidences are accumulated across events.  
- Supports remapping and subset analysis.  
- Reports mean coincidences per cell.

---

### plot_channel_adc_distribution(events, board1, board2, save_fig=False, remap=False)
Plots ADC value distributions for each channel across all events.  
- Produces histograms (log scale) of ADC counts per channel.  
- Compares both boards side by side.  
- Can save each channel’s distribution as a figure.

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

### getUpstreamVeto(events, veto_board, veto_threshold)
Applies an upstream veto cut to events.  
- Returns lists of good and vetoed events.  
- Reports number of surviving beam hits.

---

### main()
Main pipeline for hodoscope analysis. Runs:  
1. Upstream veto selection  
2. Hit detection and filtering  
3. Good event intersection (veto + hit cuts)  
4. Coincidence hitmap generation  
