#  Facial Expression Recognition MAML-Pytorch



# Platform
- python: 3.7
- Pytorch: 1.8.1


# Fer2013

## Howto
1. download fer2013 dataset from kaggle.
2. run `python preprocess_fer2013.py` to preprocess  fer2013 dataset.
3. run `python fer_maml_train.py`, the program will create MAML models in models directory.
4. run `python fer_maml_eval.py --reload_model models/maml_model.pt` to evaluate the model on test task.


