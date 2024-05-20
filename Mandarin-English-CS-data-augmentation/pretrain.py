import sys, os, logging, time, random, math, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
from torch.autograd import Variable
from torchtext.vocab import Vectors
from torchtext.data import Field, TabularDataset, BucketIterator
from models import Encoder, Decoder, Attention, Seq2Seq, Discriminator
torch.manual_seed(1)

def pretrain_G(model, iterator, optimizer, criterion, clip, A2B=True):
    
  optimizer.zero_grad()
  model.train()
    
  epoch_loss = 0
  for i, batch in enumerate(iterator):
    if A2B is True:
      src, src_len = batch.src_zh
      trg, trg_len = batch.src_cs
    else:
      src, src_len = batch.src_cs
      trg, trg_len = batch.src_zh

    output, _= model(src, src_len, trg)
    output = output[1:].view(-1, output.shape[-1])
    trg = trg[1:].view(-1)

    loss = criterion(output, trg)
        
    loss.backward()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
    optimizer.step()
        
    epoch_loss += loss.item()

    #print(f'{i} total loss: {loss}')

  return (epoch_loss) / len(iterator)

def evaluate_G(model, iterator, criterion, A2B=True):
    
  model.eval()
   
  epoch_loss = 0
    
  with torch.no_grad():

    for i, batch in enumerate(iterator):
      if A2B is True:
        src, src_len = batch.src_zh
        trg, trg_len = batch.src_cs
      else:
        src, src_len = batch.src_cs
        trg, trg_len = batch.src_zh

      output, _ = model(src, src_len, trg)
      output = output[1:].view(-1, output.shape[-1])
      trg = trg[1:].view(-1)
      loss = criterion(output, trg)
        
      epoch_loss += loss.item()

      #print(f'{i} total loss: {loss}')

  return (epoch_loss) / len(iterator)

def pretrain_D(model, iterator, optimizer, criterion, clip):
    
  optimizer.zero_grad()
  model.train()
    
  epoch_loss = 0
  for i, batch in enumerate(iterator):
    src, src_len = batch.src
    label = batch.label

    pred = model(src, src.shape[1])
    loss = criterion(pred, label)
        
    loss.backward()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
    optimizer.step()
        
    epoch_loss += loss.item()

    #print(f'{i} total loss: {loss}')

  return (epoch_loss) / len(iterator)

def evaluate_D(model, iterator, criterion):
    
  model.eval()
    
  epoch_loss = 0
    
  with torch.no_grad():
    
    for i, batch in enumerate(iterator):
      src, src_len = batch.src
      label = batch.label
            
      pred = model(src, src.shape[1]) 
      loss  = criterion(pred, label) 
            
      epoch_loss += loss.item()

      #print(f'{i} total loss: {loss}')

  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)
        
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tokenizer(sentence):
  return [ w for w in sentence.strip().split()]

def translate_sentence(model, SRC, tokenized_sentence):
  model.eval()
  tokenized_sentence = [t.lower() for t in tokenized_sentence]
  numericalized = [SRC.vocab.stoi[t] for t in tokenized_sentence] 
  sentence_length = torch.LongTensor([len(numericalized)]).to(device) 
  tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 
  translation_tensor_logits, attention = model(tensor, sentence_length, None, 0) 
  translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
  translation = [SRC.vocab.itos[t] for t in translation_tensor]
  translation, attention = translation[1:], attention[1:]
  return translation, attention

class LambdaLR():
  def __init__(self, n_epochs, offset, decay_start_epoch):
    assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
    self.n_epochs = n_epochs
    self.offset = offset
    self.decay_start_epoch = decay_start_epoch

  def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def parse():
  ap=argparse.ArgumentParser()
  ap.add_argument('--lr', type=float, default=0.0002, help='learning rate')
  ap.add_argument('--epochs', type=int, default=10, help='the number of epochs')
  ap.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
  ap.add_argument('--batchsize', type=int, default=64, help='the batch size')
  ap.add_argument('--eunits', type=int,default=650, help='the units of Encoder')    
  ap.add_argument('--enc_dropout', type=float,default=0, help='the dropout of Encoder (0-1)')    
  ap.add_argument('--enc_emb', type=int, default=300, help='the embedding of Decoder')    
  ap.add_argument('--dunits', type=int, default=650, help='the units of Decoder')    
  ap.add_argument('--dec_dropout', type=float, default=0, help='the dropout of Decoder (0-1)')    
  ap.add_argument('--dec_emb', type=int, default=300, help='the embedding of Decoder')    
  ap.add_argument('--dis_emb', type=int, default=300, help='the embedding of Discriminator')    
  ap.add_argument('--dis_units', type=int, default=650, help='the units of Discriminator (LSTM)')    
  ap.add_argument('--data', default='./train_data/AISSMS_40K', help='the directory of training data')    
  ap.add_argument('--exp_dir', default='./exp/AISSMS_40K', help='the directory of output')    
  ap.add_argument('--model', default='A2B', help='which model to be train (A2B, B2A or Discriminator)')    
  args=ap.parse_args()

  return args

if __name__ == '__main__':
  opt = parse()

  if not os.path.exists(opt.data):
    raise Exception('the directory of training data ' + opt.data + 'doesnot exist !!')
  
  if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

  ## Load Kaldi ark data

  ## Using GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  ## Networks setting ##
  FEAT_DIM=40 # fbank num-bins
  if opt.model == 'A2B' or opt.model == 'B2A':
    # models for Generator
    attn = Attention(opt.eunits, opt.dunits)
    ENC = Encoder(FEAT_DIM, opt.eunits, opt.dunits, opt.enc_dropout)
    DEC = Decoder(FEAT_DIM, opt.eunits, opt.dunits, opt.dec_dropout, attn)
    net = Seq2Seq(ENC, DEC, device).to(device)
    net.cuda()
    net.apply(init_weights)
    number1=count_parameters(net)
    print(f'The model has {number1} trainable parameters')

    # Optimizers
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    # Lossess
    criterion = nn.CrossEntropyLoss(ignore_index=G_PAD_IDX)

    best_valid_loss = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.model == 'A2B':
      # Data iterator for Generator (A->B)
      train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = opt.batchsize,
        sort_within_batch = True,
        sort_key = lambda x : len(x.src_zh),
        device = device)
      isA2B = True
    else:
      # Data iterator for Generator (B->A)
      train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = opt.batchsize,
        sort_within_batch = True,
        sort_key = lambda x : len(x.src_cs),
        device = device)
      isA2B = False

    for epoch in range(opt.epochs):
      start_time = time.time()

      print('pre-train ' + opt.model)
      train_loss = pretrain_G(net, train_iterator, optimizer, criterion, opt.clip, isA2B)
      valid_loss = evaluate_G(net, train_iterator, criterion, isA2B)

      end_time = time.time()

      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        filename = opt.exp_dir + '/' + str(epoch)+'_bs' + str(opt.batchsize) + '_net' + str(opt.model) + '.pt'
        torch.save(net.state_dict(), filename)
    
      print(f'[Pretrain] Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\t[Pretrain] Train Loss: {train_loss:.3f}')
      print(f'\t[Pretrain] Val. Loss: {valid_loss:.3f}')
    
    # Print the prediction
    # Load PKUSMS data to field SRC_ZH, SRC_EN and SRC_CS
    train_PKUSMS_data, valid_PKUSMS_data= TabularDataset.splits(
      path=opt.data, train='Train_smsCorpus_zh.seg.pku.CS.json',  validation='Valid_smsCorpus_zh.seg.pku.CS.json', format='json',
      fields={"src_zh": ("src_zh", SRC_ZH), "src_cs": ("src_cs", SRC_CS)})

    prediction = opt.exp_dir + '/' + opt.model + '_prediction_test'
    with open(prediction, 'w') as f:
      for i in range(len(test_data.examples)):
        if opt.model == 'A2B':
          src = vars(test_data.examples[i])['src_zh']
          trg = vars(test_data.examples[i])['src_cs']
        else:
          src = vars(test_data.examples[i])['src_cs']
          trg = vars(test_data.examples[i])['src_zh']

        print(f'src = {src}')
        print(f'trg = {trg}')
   
        if opt.model == 'A2B':
          translation, attention = translate_sentence(net, SRC_ZH, src)
        else:
          translation, attention = translate_sentence(net, SRC_CS, src)
      
        print(f'predicted trg = {translation}')
        sentence = ' '.join(translation)
        print(sentence+'\n')
        f.write(sentence+'\n')  
    print('Save the predictions of test to ' + prediction)  

    prediction = opt.exp_dir + '/' + opt.model + '_prediction_PKUSMS_valid'
    with open(prediction, 'w') as f:
      for i in range(len(valid_PKUSMS_data.examples)):
        if opt.model == 'A2B':
          src = vars(valid_PKUSMS_data.examples[i])['src_zh']
          trg = vars(valid_PKUSMS_data.examples[i])['src_cs']
        else:
          src = vars(valid_PKUSMS_data.examples[i])['src_cs']
          trg = vars(valid_PKUSMS_data.examples[i])['src_zh']

        print(f'src = {src}')
        print(f'trg = {trg}')
   
        if opt.model == 'A2B':
          translation, attention = translate_sentence(net, SRC_ZH, src)
        else:
          translation, attention = translate_sentence(net, SRC_CS, src)

        print(f'predicted trg = {translation}')
        sentence = ' '.join(translation)
        print(sentence+'\n')
        f.write(sentence+'\n')
    print('Save the prediction of PKUSMS_valid to ' + prediction)  

  else:
    # models for D_A and D_B
    netD_A = Discriminator(DA_INPUT_DIM, DA_OUTPUT_DIM, opt.batchsize, opt.dis_units, opt.dis_emb)
    netD_B = Discriminator(DB_INPUT_DIM, DB_OUTPUT_DIM, opt.batchsize, opt.dis_units, opt.dis_emb)
    netD_A.cuda()
    netD_B.cuda()
    netD_A.apply(init_weights)
    netD_B.apply(init_weights)
    number2=count_parameters(netD_A)
    number3=count_parameters(netD_B)
    print(f'The model has {number2} trainable parameters')
    print(f'The model has {number3} trainable parameters')
    
    # Optimizers
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Lossess
    criterion = nn.CrossEntropyLoss(ignore_index=DA_PAD_IDX)

    best_DA_valid_loss = float('inf')
    best_DB_valid_loss = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data iterator for Discriminator A
    train_DA_iterator, valid_DA_iterator = BucketIterator.splits(
      (train_DA_data, valid_DA_data), 
      batch_size = opt.batchsize,
      sort_within_batch = True,
      sort_key = lambda x : len(x.src),
      device = device)

    # Data iterator for Discriminator B
    train_DB_iterator, valid_DB_iterator = BucketIterator.splits(
      (train_DB_data, valid_DB_data), 
      batch_size = opt.batchsize,
      sort_within_batch = True,
      sort_key = lambda x : len(x.src),
      device = device)

    for epoch in range(opt.epochs):
      start_time = time.time()
    
      print('pre-train DA')
      train_DA_loss = pretrain_D(netD_A, train_DA_iterator, optimizer_D_A, criterion, opt.clip)
      valid_DA_loss = evaluate_D(netD_A, valid_DA_iterator, criterion)
      print('pre-train DB')
      train_DB_loss = pretrain_D(netD_B, train_DB_iterator, optimizer_D_B, criterion, opt.clip)
      valid_DB_loss = evaluate_D(netD_B, valid_DB_iterator, criterion)
    
      end_time = time.time()
    
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
      if valid_DA_loss < best_DA_valid_loss:
        best_DA_valid_loss = valid_DA_loss
        torch.save(netD_A.state_dict(), opt.exp_dir+'/'+str(epoch)+'_bs'+str(opt.batchsize)+'_netD_A.pt')
    
      if valid_DB_loss < best_DB_valid_loss:
        best_DB_valid_loss = valid_DB_loss
        torch.save(netD_B.state_dict(), opt.exp_dir+'/'+str(epoch)+'_bs'+str(opt.batchsize)+'_netD_B.pt')
    
      print(f'[Pretrain] Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\t[Pretrain] Train Loss: {train_DA_loss:.3f}')
      print(f'\t[Pretrain] Val. Loss: {valid_DA_loss:.3f}')
      print(f'\t[Pretrain] Train Loss: {train_DB_loss:.3f}')
      print(f'\t[Pretrain] Val. Loss: {valid_DB_loss:.3f}')

