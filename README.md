#Overview
This code integrates a question classifier system with the convolutional neural network architecture for learning to match question and answer sentences implemented by Aliaksei Severyn. The original repository of the question answer system can be found [here](https://github.com/aseveryn/deep-qa).

#Requirement

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

The above package if not installed, can be installed using pip command for python : pip install <package-name>

#Embedding
The pre-trained embeddings used in the QA system can be downloaded [here](https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing)

#Build
To build the required train/dev/test sets in the suitable format for the network run:

>$ sh run_build_datasets.sh

Download the word-embeddings from above link and place it in the folder named 'embeddings'

#Deployment - without Question Classifier
To train the model in the TRAIN setting run:

>$ python run_nnet.py TRAIN

in the TRAIN-ALL setting using 53,417 qa pairs:

>$ python run_nnet.py TRAIN-ALL

#Deployment - with Question Classifier
Download the pre-trained Question Classification models from [here](https://drive.google.com/open?id=0B11zdsTNhzfGcDVXYmkwNXBST28).

The question classifier contains a different vocabulary and embedding, but since the input is already passed as the vocab_index and not as word in the previous model, we convert the embedding of the classifier to match the vocabulary of the existing system.

>$ python convert_embeddings.py

This wil create appropriate embeddings for each of the previous embddings in the models folder.

To train the model with this question classifier models

>$ python run_nnet.py <train_data> <trained_model_path> <network_the_model_was_trained_on>

trained_data : TRAIN or TRAIN-ALL
trained_model_path : QC_models/TREC/LSTM
network_the_model_was_trained_on : LSTM or GRU