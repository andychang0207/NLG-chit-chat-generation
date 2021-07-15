
# ADL-FINAL NLG chitchat




## Installation 



```bash 
pip install -r requirements.txt
```
    
## Training

To run training with response in front of chitchat, run the following command

```bash
bash train_script_res.sh data/path/to/train.json data/path/to/eval.json
```

To run training with chitchat in front of response, run the following command

```bash
bash train_script.sh data/path/to/train.json data/path/to/eval.json
```


## Generation

To run generation with golden response, run the following command

```bash
bash generate_script_res.sh data/path/to/test.json data/path/to/prediction.json
```

To run generation with predicted response, run the following command

```bash
bash generate_script.sh data/path/to/test.json data/path/to/prediction.json
```