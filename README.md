# LEGO Model Data Processing (Step 1)

This step of the project demonstrates how to process LEGO models (in
**MPD/LDR format**) into structured DataFrames.

## Pipeline Overview

1.  **Load model file** → Import `.mpd`/`.ldr` into a raw DataFrame
    (model structure).\
2.  **Extract part numbers** → Get part IDs from the raw DataFrame.\
3.  **Fetch part info** → Enrich parts with metadata from the
    Rebrickable API.\
4.  **Merge category data** → Add part categories to the DataFrame.\
5.  **Extract structural info** → Functions applied to `part_name` for
    dimensions and attributes.\
6.  **Export results** → Two CSVs per model:
    -   `<model_name>_parts.csv` (parts metadata)\
    -   `<model_name>_model.csv` (model dataframe)

This provides the foundation for further analysis and visualization in
later steps.
