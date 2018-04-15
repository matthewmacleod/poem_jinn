"""
to run:
    python src/main.py
"""
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus
import re
import sys
import argparse

from pathlib import Path

from models import LSTMLM

def main():
    parser = argparse.ArgumentParser(description="RNN language model training")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--model_dir", default="models/")
    parser.add_argument("--target_text", help='txt file in data',
                        type=str, default="shelley")
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--use_gpu", help='0 false 1 true', type=int, default=0)
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--burn", type=int, default=5)

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
    
    # Load Dataset
    train_name = re.sub(args.data_dir, '', train_path)
    train_name = re.sub('.txt', '', train_name)
    
    corpus = Corpus()
    ids = corpus.get_data(train_path, args.batch_size)
    vocab_size = len(corpus.dictionary)
    print("Total vocabulary size:", vocab_size)
    num_batches = ids.size(1) // args.seq_length
    
    model = LSTMLM(vocab_size, args.embed_size,
                   args.hidden_size, args.num_layers, args.dropout)
    
    if use_gpu:
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Truncated Backpropagation 
    def detach(states):
        return [state.detach() for state in states] 
    
    # Train
    for epoch in range(args.num_epochs):
        # Initial hidden and memory states # wrap them in Variable
        if use_gpu:
        	states = (Variable(torch.zeros(args.num_layers,
                                           args.batch_size, args.hidden_size)).cuda(),
                      Variable(torch.zeros(args.num_layers, args.batch_size,
                                           args.hidden_size)).cuda())
        else:
        	states = (Variable(torch.zeros(args.num_layers,
                                           args.batch_size, args.hidden_size)),
                      Variable(torch.zeros(args.num_layers,
                                            args.batch_size, args.hidden_size)))
        
        for i in range(0, ids.size(1) - args.seq_length, args.seq_length):
            # Get batch inputs and targets
            if use_gpu:
                inputs = Variable(ids[:, i:i+args.seq_length]).cuda()
                targets = Variable(ids[:, (i+1):(i+1)+args.seq_length].contiguous()).cuda()
            else:
                inputs = Variable(ids[:, i:i+args.seq_length])
                targets = Variable(ids[:, (i+1):(i+1)+args.seq_length].contiguous())
            
            # Forward + Backward + Optimize
            model.zero_grad()
            states = detach(states)
            outputs, states = model(inputs, states) 
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
    
            step = (i+1) // args.seq_length
            if step % 100 == 0:
                print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                       (epoch+1, args.num_epochs, step, num_batches,
                       loss.data[0], np.exp(loss.data[0])))
    
    # Save the Trained Model
    torch.save(model.state_dict(), args.model_dir + 'model_' +
               train_name + '_exp_' + str(args.exp) + '.pkl')
    print('Model training finished!')
    return

if __name__ == "__main__":
    main()
