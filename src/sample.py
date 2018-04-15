import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus
import re
import sys
from pathlib import Path
import argparse

from models import LSTMLM


def main():
    parser = argparse.ArgumentParser(description="RNN language model training")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--model_dir", default="models/")
    parser.add_argument("--sample_dir", default="samples/")
    parser.add_argument("--target_text",
                        help='text file in data to learned from',
                        type=str, default="shelley")
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--use_gpu", help='0 false 1 true', type=int, default=0)
    parser.add_argument("--exp", type=int, default=0)

    args = parser.parse_args()
    print('Args:', args)

    can_use_gpu = torch.cuda.is_available()
    if args.use_gpu == 1:
        if not can_use_gpu:
            sys.exit('Cannot find gpu device')
        else:
            use_gpu = True  
    else:
            use_gpu = False   
    
    print('Target text:', args.target_text)
    train_path = args.data_dir + args.target_text + '.txt'
    my_file = Path(train_path)
    if not my_file.is_file():
        sys.exit('Missing file:', my_file)

    # load Dataset
    train_name = re.sub(args.data_dir,'', train_path)
    train_name = re.sub('.txt', '', train_name)
    sample_path = args.sample_dir + 'sample_' + train_name + '_exp_' + str(args.exp) + '.txt'

    corpus = Corpus()
    ids = corpus.get_data(train_path, args.batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // args.seq_length

    model = LSTMLM(vocab_size, args.embed_size,
                   args.hidden_size, args.num_layers, args.dropout)

    # Load Trained Model
    model.load_state_dict(torch.load(args.model_dir + 'model_' +
                                     train_name + '_exp_' + str(args.exp) +
                                     '.pkl'))

    # Place in inference mode (eg, no dropout)
    model.eval()

    # Sampling
    with open(sample_path, 'w') as f:
        # Set intial hidden ane memory states
        state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
                 Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
    
        # Select one word id randomly to start
        prob = torch.ones(vocab_size)
        input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),
                         volatile=True)
    
        for i in range(args.num_samples):
            # Forward propagate rnn 
            output, state = model(input, state)
            
            # Sample a word id according to our probabilities
            prob = output.squeeze().data.exp()
            word_id = torch.multinomial(prob, 1)[0]
            
            # Feed sampled word id to next time step
            input.data.fill_(word_id)
            
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)
    
            if (i+1) % 100 == 0:
                print('Sampled [%d/%d] words and save to %s'%(i+1, args.num_samples,
                                                              sample_path))

    return

if __name__ == "__main__":
    main()
