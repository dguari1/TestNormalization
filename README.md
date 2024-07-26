This repository is used to re-normalize the finger-taping results provided by the Vision PD tool. 

The JSON files provided by the tool should be uploaded to the `data` folder. This repository can be used to process multiple JSON files.

After all the JSON files are available, execute 

```
python process.py
```

This will generate `.csv` files in the folder `output` containing the measurements with the new normalization factor. 

There are two possible normalization factors 
1. **THUMBSIZE** 
3. INDEXSIZE

THUMBSIZE is the default normalization factor. To change the normalization factor, open `process.py` and in line 199, change THUMBSIZE to INDEX
