# Overview
This code integrates a question classifier system with the convolutional neural network architecture for learning to match question and answer sentences implemented by Aliaksei Severyn. The original repository of the question answer system can be found <a href = 'https://github.com/aseveryn/deep-qa' target='_blank'> here</a>.

# Requirement

- python 2.7+
- numpy
- scikit-learn (sklearn)
- tensorflow
- theano
- keras
- pandas
- tqdm
- fish
- numba

The above package if not installed, can be installed using pip command for python : `pip install <package-name>`

#  Embedding
The pre-trained embeddings used in the QA system can be downloaded <a href = 'https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing' target='_blank'> here</a>.

# Build
Download the word-embeddings from above link and place it in the folder named 'embeddings'.

To build the required train/dev/test sets in the suitable format for the network run:

>$ sh run_build_datasets.sh

# Deployment - without Question Classifier
To train the model in the TRAIN setting run:

>$ python run_nnet.py TRAIN

in the TRAIN-ALL setting using 53,417 qa pairs:

>$ python run_nnet.py TRAIN-ALL

The parameters of the trained network are dumped under the 'exp.out' folder.  
TRAIN:
MAP: 0.7325
MRR: 0.8018

TRAIN-ALL:
MAP: 0.7538
MRR: 0.8078

# Deployment - with Question Classifier
Download the pre-trained Question Classification models from <a href = 'https://drive.google.com/open?id=0B11zdsTNhzfGVzd5WXQzUTJ1cDg' target='_blank'> here</a>.
The folder contains the pre-trained question classifier models. It contains three folders namely vocab, TREC and MT.  

QC_models/vocab - contains the vocabulary files used to train the models  
QC_models/TREC - contains pre-trained model trained on TREC data only   
QC_models/MT - contains pre-trained model trained using multitask learning  

The question classifier contains a different vocabulary and embedding, but since the input is already passed as the vocab_index and not as word in the previous model, we convert the embedding of the classifier to match the vocabulary of the existing system.

>$ python convert_embeddings.py

This wil create appropriate embeddings for each of the previous embddings in the models folder.

To train the model with this question classifier models

```sh
python run_nnet.py <train_data> <trained_QC_model_path> <network_QC__was_trained_on>
```

example :
>$ python run_nnet.py TRAIN QC_models/TREC/LSTM/ LSTM

train_data : TRAIN or TRAIN-ALL

trained_QC_model_path : the location of the pre-trained model where the embedding is present as well.  

network_QC_was_trained_on : LSTM or GRU

# Best result:

>$ python run_nnet.py TRAIN-ALL QC_models/MT/LSTM/ LSTM

TRAIN:
MAP: 0.7452
MRR: 0.8080

TRAIN-ALL:
MAP: 0.7779
MRR: 0.8093