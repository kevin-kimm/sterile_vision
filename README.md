# sterile_vision
Detection of biocontaminants / blood residue on surgical instruments for sterile processing

## Dataset
This project uses the following dataset as its base:

**SURGICAL TOOLS** — Mendeley Data  
- Author: Sravan Reddy  
- Published: March 17, 2025  
- DOI: 10.17632/cyghvmjrt3.1  
- Link: https://data.mendeley.com/datasets/cyghvmjrt3/1  
- License: CC BY-ND 4.0  

6,000 high quality images across 9 surgical instrument categories captured under diverse conditions including artificial blood contamination, varying lighting, and 360 degree angles.

## Contamination Augmentation
Since the original dataset was designed for tool recognition rather than contamination detection, synthetic blood/biocontaminant stains were applied to clean instrument images using a custom augmentation pipeline (`scripts/augment_contamination.py`). Real contaminated images from the dataset's "Overlapping" category were also incorporated.
