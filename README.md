This repository is used to re-normalize the finger-taping results provided by the Vision PD tool. 

The JSON files provided by the tool should be uploaded to the `data` folder. This repository can be used to process multiple JSON files.

After all the JSON files are available, execute 

```
python process.py
```

This will generate `.csv` files in the folder `output` containing the measurements with the new normalization factor. 

There are three options:
1. **'THUMBSIZE'**
2. 'INDEXSIZE'
3. 'NOSCALING' 

THUMBSIZE is the default normalization factor. To change the normalization factor, open `process.py` and change the variable ``scalingMethod`` in ``main()``. 

NOSCALING will use the same scaling factor currently available in the JSON file. 
