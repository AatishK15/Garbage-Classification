# Dataset Information

## Source

- **Name**: Garbage Classification Dataset (Extended)
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **License**: CC0: Public Domain / Open Data
- **Original Work**: Based on TrashNet by Gary Thung and Mindy Yang (Stanford, 2016)

---

## Dataset Details

### Original 12 Classes

The Kaggle dataset contains 12 categories:

| Class | Approximate Count |
|-------|-------------------|
| Battery | ~500 |
| Biological | ~900 |
| Brown Glass | ~600 |
| Cardboard | ~400 |
| Clothes | ~1,000 |
| Green Glass | ~500 |
| Metal | ~800 |
| Paper | ~1,000 |
| Plastic | ~900 |
| Shoes | ~600 |
| Trash | ~300 |
| White Glass | ~500 |

**Total**: ~15,000+ images

### Mapping to 7 Target Classes

We consolidate the 12 classes into 7 semantically meaningful categories:

| Target Class | Source Classes | Rationale |
|-------------|---------------|-----------|
| **Cardboard** | cardboard | Direct mapping |
| **Glass** | brown-glass, green-glass, white-glass | All are glass recyclables |
| **Metal** | metal, battery | Both are metal-based items |
| **Organic** | biological | Organic/compostable waste |
| **Paper** | paper | Direct mapping |
| **Plastic** | plastic | Direct mapping |
| **Trash** | trash, clothes, shoes | Non-recyclable / mixed items |

---

## Preprocessing Pipeline

1. **Image Collection**: Downloaded from Kaggle and organized into target class directories
2. **Resizing**: All images resized to 224×224 pixels (MobileNetV2 input size)
3. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
4. **Data Split**: Stratified 70/15/15 train/val/test split
5. **Augmentation** (training only):
   - Random rotation: ±20°
   - Width/height shift: ±20%
   - Shear: ±15%
   - Zoom: ±15%
   - Horizontal flip: Yes
   - Brightness: 0.8–1.2×

---

## Class Imbalance

The dataset exhibits moderate class imbalance (e.g., Glass has more images due to 3 source classes). This is handled by:

1. **Inverse frequency class weights** computed via scikit-learn
2. **Data augmentation** to synthetically expand minority classes
3. **Stratified splitting** to maintain class proportions across all splits

---

## File Formats

- **Image formats**: JPEG, PNG
- **Image sizes**: Variable (resized to 224×224 during preprocessing)
- **Color space**: RGB (3 channels)

---

## Citation

If using this dataset, please cite:
- Thung, G., & Yang, M. (2016). Classification of Trash for Recyclability Status. CS229 Project Report, Stanford University.
- Kaggle Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
