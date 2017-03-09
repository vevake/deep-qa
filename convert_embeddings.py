import numpy as np
import cPickle
import os, json

from alphabet import Alphabet
from utils import load_bin_vec


path_all = ['TREC/LSTM/', 'TREC/GRU', 'MT/LSTM', 'MT/GRU']

def main():
  for path in path_all :
    np.random.seed(123)

    data_dirs = [
                'TRAIN',
                'TRAIN-ALL',
                ]
    if 'TREC' in path:
      v = json.loads(open('QC_models/vocab/vocab_TREC.json').read())
    elif 'MS' in path:
      v = json.loads(open('QC_models/vocab/vocab_MS.json').read())
    else :
      print 'Error loading vocabulary'
      sys.exit(1)
    words = v.keys()    
    i =0
    embedding_weights = np.load('QC_models/' + path + 'embedding.npy')
    found = 0
    for data_dir in data_dirs:
      fname_vocab = os.path.join(data_dir, 'vocab.pickle')
      alphabet = cPickle.load(open(fname_vocab))
      a = alphabet[max(alphabet, key = alphabet.get)] + 1 
      vocab_emb = np.zeros((len(alphabet) + 1, 300))
      for word, idx in alphabet.iteritems():
        if word in v.keys():     
          word_vec = embedding_weights[v[word]]
          vocab_emb[idx] = word_vec
          found += 1
        else :
          vocab_emb[idx] = embedding_weights[1]
      vocab_emb[a] = embedding_weights[0]                    
      i +=1
      #print 'found :', found
      outfile = 'QC_models/' + path + 'emb_{}.npy'.format(data_dir)
      print outfile + ' completed'
      np.save(outfile, vocab_emb)


if __name__ == '__main__':
  main()
