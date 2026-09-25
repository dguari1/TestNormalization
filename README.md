This repository is used to re-normalize the finger-taping results provided by the Vision PD tool. 

The JSON files provided by the tool should be uploaded to the `data` folder. This repository can be used to process multiple JSON files.

After all the JSON files are available, execute 

```
python process.py
```

This will generate `.csv` files in the folder `output` containing the measurements with the new normalization factor. 

There are three options:
1. `NOSCALING`
2. `THUMBSIZE`
3. `INDEXSIZE`
4. `HANDSPAN` #to implement
5. `PALMSIZE` #to implement

'NOSCALING' is the default value. To change the normalization factor, open `process.py` and change the variable ``scalingMethod`` in ``main()``. 

- `NOSCALING` will use the same scaling factor currently available in the JSON file.
- `INDEXSIZE` will use the size of the INDEX finger (estimated from the provided landmarks) as the normalization factor 
- `THUMBSIZE` will use the size of the THUMB finger (estimated from the provided landmarks) as the normalization factor 
- `HANDSPAN` will use the size of the hand span (distance from the wrist to the tip of the index finger, estimated from the provided landmarks) as the normalization factor 
- `PALMSIZE` will use the size of the palm (estimated from the provided landmarks) as the normalization factor

## Technical quality checks

Each output CSV includes a `quality_check` row (`Good`, `Needs review`, or
`Failed`) plus auditable `quality_*` metrics and reasons. The checks are the
signal-based VisionMD technical-quality screen: sample count, invalid values,
flat signals, and isolated jumps. They flag technical concerns only and do not
establish clinical validity. The generated plot shows the same quality label,
reasons, and a green/amber/red background.

Archived JSON files do not include live P/S pipeline diagnostics, so the
P/S-specific VisionMD quality checks cannot be recreated in this batch tool.
