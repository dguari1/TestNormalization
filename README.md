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
