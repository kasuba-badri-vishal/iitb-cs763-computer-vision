[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/m05MzDli)
# Group-06 CV Project

## Handwritten Word Recognition for Indian Languages using CRNN model

### Project Demo Video

https://user-images.githubusercontent.com/40017320/232036494-69e5182b-3aef-4f4c-bb37-90a4c61bd77e.mp4

Youtube link of Demo - [link](https://www.youtube.com/watch?v=qhZOs9r7fLY)

### Procedure

1. Obtaining the Dataset from [IHTR website](https://ilocr.iiit.ac.in/ihtr/dataset.html)
2. Preprocessing the dataset and organizing data for training
3. Creating Vocabulary set from the dataset for training respective characters of languages
4. Training the Model (we have trained upto 100 epochs for each language)
5. Evaluating and Testing upon the trained models to get predictions

### Train Command

##### Training the Model uses opensource library DocTR, through which CRNN model is trained from scratch. The path to training data and validation data must be provided along with other hyperparameters


```
python doctr/references/recognition/train_pytorch.py crnn_vgg16_bn --train_path ./data/tamil/train/ --val_path /data/BADRI/IHTR/validationset_small/tamil/ --vocab tamil --epochs 100 --b 1024 --device 0 --lr 0.01
```

### Test Command

```
python src/test.py --data path/to/test/data/ --lang telugu --model ./models/crnn_vgg16_bn_telugu.pt
```

Some test data is present as test_samples folder that is shared just for reference

### Get Evalaution Results

```
python src/get_results.py --data path/to/validation/data/ --lang telugu --model ./models/crnn_vgg16_bn_telugu.pt
```
