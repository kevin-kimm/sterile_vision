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

Final dataset breakdown:
- Train: 3,136 clean | 4,200 contaminated (1,064 real + 3,136 synthetic)
- Val:     896 clean | 1,200 contaminated
- Test:    448 clean |   600 contaminated

## Contamination Augmentation
Since the original dataset was designed for tool recognition rather than contamination detection, synthetic blood/biocontaminant stains were applied to clean instrument images using a custom augmentation pipeline (`scripts/augment_contamination.py`). Real contaminated images from the dataset's "Overlapping" category were also incorporated.

## Synthetic Contamination Methodology

Since real labeled images of contaminated surgical instruments from sterile 
processing departments (SPDs) are not publicly available, this project uses 
a synthetic augmentation approach to simulate biocontaminant residue.

### How It Works
The `augment_contamination.py` script applies programmatic blood/tissue stain 
overlays to clean instrument images using OpenCV. Each synthetic stain is 
generated with:

- **Irregular blob shapes** — using random polygon generation to simulate 
  organic splatter patterns
- **Realistic color palette** — three blood states are simulated:
  - Fresh blood (bright red: RGB 190, 30, 20)
  - Dried blood (dark red/brown: RGB 140, 45, 15)
  - Old dried blood (dark brown: RGB 90, 30, 10)
- **Splatter drops** — small satellite droplets around each main blob
- **Occasional smear effects** — simulating tool-to-surface contact marks
- **Randomized placement** — stains are focused on the center 60% of the 
  image where the tool is located

### Real World Application
In a real clinical deployment, this model would be retrained on actual images 
of surgical instruments collected at the point of decontamination in an SPD. 
SPD technicians would photograph instruments before cleaning, and a labeled 
dataset of clean vs contaminated instruments would replace the synthetic data. 

The synthetic approach used here serves as a proof of concept demonstrating 
that:
1. The pipeline architecture is sound
2. The model can learn contamination features effectively
3. With real SPD imagery the same approach could achieve clinical accuracy

A partnership with a hospital SPD department and IRB approval for image 
collection would be the logical next step toward real world deployment.

## Model
- Architecture: YOLOv8n-cls (nano classification)
- Training: 29 epochs (early stopped)
- Training time: ~20 minutes on Apple M4
- Final accuracy: 99.90% on test set (1,047/1,048 correct)

## Scripts

### 1. Prepare Dataset
Splits the raw Mendeley dataset into train/val/test:
```bash
python scripts/prepare_dataset.py
```

### 2. Augment Contamination
Generates synthetic contamination and builds the final dataset:
```bash
python scripts/augment_contamination.py
python scripts/flatten_dataset.py
```

### 3. Train
Trains the YOLOv8 classification model:
```bash
python scripts/train.py
```

### 4. Evaluate
Runs the model on the full test set and saves results to `outputs/logs/`:
```bash
python scripts/evaluate.py
python scripts/save_results.py
```

### 5. Predict
Tests the model on a single clean vs contaminated image pair:
```bash
python scripts/predict.py
```

### 6. Test Real World Images
Tests the model on real world images from outside the dataset.
Automatically generates contaminated versions using the same synthetic 
augmentation pipeline used in training.

Place clean images in `data/test_images/clean/` and run:
```bash
python scripts/test_real_world.py
```

Results on 5 real world images (10 total predictions — clean + contaminated):
- episiomity_01.jpeg  → clean 99.96% PASS | contaminated 100.00% PASS
- forceps_01.jpg      → clean 99.69% PASS | contaminated 100.00% PASS
- hemostat_01.png     → clean 99.98% PASS | contaminated 100.00% PASS
- scalpel_01.jpeg     → clean 99.96% PASS | contaminated  99.35% PASS
- scissors_01.png     → clean 99.78% PASS | contaminated  96.71% PASS

**Total: 10/10 correct (100.00%)**

## Limitations

- Synthetic contamination does not perfectly replicate real biocontaminants
- Model is sensitive to background color (trained on blue surgical drapes)
- Real rust, biofilm, and dried blood look different from synthetic stains
- A real world SPD dataset with genuine contamination labels would be needed
  for clinical deployment