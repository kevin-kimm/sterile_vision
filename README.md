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

### 6. Test New Images
Tests the model on real world images from outside the dataset.
Place images in `data/test_images/` and run:
```bash
python scripts/test_new_images.py
```

Results on 3 real world forceps images:
- Clean forceps white background → CLEAN 99.90% PASS
- Clean forceps green background → CONTAMINATED 100% FAIL
- Rusty contaminated forceps → CLEAN 99.63% FAIL

The model performs excellently on synthetic contamination (99.9%) but struggles 
to generalize to real world contamination, highlighting the need for real labeled 
SPD imagery for clinical deployment. Background color also plays a significant 
role as the model was trained primarily on blue surgical drapes.

## Limitations

- Synthetic contamination does not perfectly replicate real biocontaminants
- Model is sensitive to background color (trained on blue surgical drapes)
- Real rust, biofilm, and dried blood look different from synthetic stains
- A real world SPD dataset with genuine contamination labels would be needed
  for clinical deployment