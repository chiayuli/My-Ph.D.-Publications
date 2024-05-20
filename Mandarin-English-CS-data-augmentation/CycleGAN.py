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

def train(model, netD_A, netD_B, iterator, iteratorB,  optimizer,  clip, lam_A2B2A, lam_B2A2B, ig_index):
  criterion = nn.CrossEntropyLoss(ignore_index=ig_index)
  optimizer.zero_grad()
  model.train()
    
  epoch_loss = 0
  for i, (batch, batchB) in enumerate(zip(iterator, iteratorB)):
    src, src_len = batch.src_zh
    trg, trg_len = batch.src_cs
    src = src.transpose(0,1)
    trg = trg.transpose(0,1)
    srcB, srcB_len = batchB.src_zh
    trgB, trgB_len = batchB.src_cs
    srcB = srcB.transpose(0,1)
    trgB = trgB.transpose(0,1)
    
    loss_DA, loss_DB, loss_A2B2A, loss_B2A2B = model(src, src_len, trg, trg_len, srcB, srcB_len, trgB, trgB_len, netD_A, netD_B)
    loss = loss_DA + loss_DB + loss_A2B2A*lam_A2B2A + loss_B2A2B*lam_B2A2B    
    loss.backward()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
    optimizer.step()
    print('    ====== iterator '+str(i)+', loss='+str(loss))   
    epoch_loss += loss.item()

  return (epoch_loss) / len(iterator)

def evaluate(model, netD_A, netD_B, iterator, iteratorB, lam_A2B2A, lam_B2A2B):
    
  model.eval()
  epoch_loss = 0
    
  with torch.no_grad():

    for i, (batch, batchB) in enumerate(zip(iterator, iteratorB)):
        src, src_len = batch.src_zh
        trg, trg_len = batch.src_cs
        src = src.transpose(0,1)
        trg = trg.transpose(0,1)
        srcB, srcB_len = batchB.src_zh
        trgB, trgB_len = batchB.src_cs
        srcB = srcB.transpose(0,1)
        trgB = trgB.transpose(0,1)
    
        loss_DA, loss_DB, loss_A2B2A, loss_B2A2B = model(src, src_len, trg, trg_len, srcB, srcB_len, trgB, trgB_len, netD_A, netD_B)
        #loss = loss_DA + loss_DB + loss_A2B2A*lam_A2B2A + loss_B2A2B*lam_B2A2B    
        loss = loss_A2B2A*lam_A2B2A + loss_B2A2B*lam_B2A2B    
        
        epoch_loss += loss.item()

  return (epoch_loss) / len(iterator)

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

def generate_prediction(model, data, filename):
  with open(filename, 'w') as f:
    for i in range(len(data.examples)):
      if True: #opt.model == 'A2B':
        src = vars(data.examples[i])['src_zh']
        trg = vars(data.examples[i])['src_cs']
      else:
        src = vars(data.examples[i])['src_cs']
        trg = vars(data.examples[i])['src_zh']

      print(f'src = {src}')
      print(f'trg = {trg}')
   
      if True: #opt.model == 'A2B':
        translation, attention = translate_sentence(model, SRC_ZH, src)
      else:
        translation, attention = translate_sentence(model, SRC_CS, src)
    
      print(f'predicted trg = {translation}')
      sentence = ' '.join(translation)
      print(sentence+'\n')
      f.write(sentence+'\n')

class CCG(nn.Module):
    def __init__(self, netG_A2B, netG_B2A, netD_A, netD_B, idx):
        super().__init__()
    
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.criterion = nn.CrossEntropyLoss(ignore_index=idx)

    def forward(self, src, src_len, trg, trg_len, srcB, srcB_len, trgB, trgB_len, netD_A, netD_B):
        src = src.transpose(0,1)
        trg = trg.transpose(0,1)
        srcB = srcB.transpose(0,1)
        trgB = trgB.transpose(0,1)
        if src.shape[1] > 1:
            output_B, _ = self.netG_A2B(src, src_len, trg)
            fake_B = torch.argmax(output_B.squeeze(1), 2).to(src.device)
            fake_B_len = torch.tensor([fake_B.shape[0]]*fake_B.shape[1]).to(src.device)
            hyp_A, _ = self.netG_B2A(fake_B, fake_B_len, src)
        else:
            output_B, _ = self.netG_A2B(src, src_len, trg)
            fake_B = torch.argmax(output_B.squeeze(1), 1).to(src.device)
            fake_B = fake_B.unsqueeze(1)
            fake_B_len = torch.tensor([fake_B.shape[0]]).to(src.device)
            hyp_A, _ = self.netG_B2A(fake_B, fake_B_len, src)
    
        hyp_A = hyp_A[1:].view(-1, hyp_A.shape[-1])
        ref_A = src[1:].contiguous().view(-1).to(src.device)
        loss_identity_A2B2A = self.criterion(hyp_A, ref_A)
        #print('[CCG]loss_identity_A2B2A '+str(loss_identity_A2B2A))
        if src.shape[1] > 1:
            output_A, _ = self.netG_B2A(trgB, trgB_len, srcB)
            fake_A = torch.argmax(output_A.squeeze(1), 2).to(src.device)
            fake_A_len = torch.tensor([fake_A.shape[0]]*fake_A.shape[1]).to(src.device)
            hyp_B, _ = self.netG_A2B(fake_A, fake_A_len, trgB)
            # for DA, DB
            fake_B2 = torch.argmax(hyp_B.squeeze(1), 2).to(src.device)
            output_DA = netD_A(fake_A, fake_A.shape[1])
            false_DA = torch.tensor([0]*output_DA.shape[0]).to(src.device)
            output_DB = netD_B(fake_B2, fake_B2.shape[1])
            false_DB = torch.tensor([0]*output_DB.shape[0]).to(src.device)
        else:
            output_A, _ = self.netG_B2A(trgB, trgB_len, srcB)
            fake_A = torch.argmax(output_A.squeeze(1), 1)
            fake_A = fake_A.unsqueeze(1)
            fake_A_len = torch.tensor([fake_A.shape[0]]).to(src.device)
            hyp_B, _ = self.netG_A2B(fake_A, fake_A_len, trgB)
            # for DA, DB
            fake_B2 = torch.argmax(hyp_B.squeeze(1), 1)
            fake_B2 = fake_B2.unsqueeze(1)
            output_DA = netD_A(fake_A, fake_A.shape[1])
            false_DA = torch.tensor([0]*output_DA.shape[0]).to(src.device)
            output_DB = netD_B(fake_B2, fake_B2.shape[1])
            false_DB = torch.tensor([0]*output_DB.shape[0]).to(src.device)
        hyp_B = hyp_B[1:].view(-1, hyp_B.shape[-1])
        ref_B = trgB[1:].contiguous().view(-1).to(src.device)
        loss_identity_B2A2B = self.criterion(hyp_B, ref_B)
        #print('[CCG]loss_identity_B2A2B '+str(loss_identity_B2A2B))
        loss_DA = self.criterion(output_DA, false_DA)
        loss_DB = self.criterion(output_DB, false_DB)
        #print('[CGG]loss_DA '+str(loss_DA)+', loss_DB '+str(loss_DB))       
        return loss_DA, loss_DB, loss_identity_A2B2A, loss_identity_B2A2B


def parse():
  ap=argparse.ArgumentParser()
  ap.add_argument('--lr', type=float, default=0.0002, help='learning rate')
  ap.add_argument('--start_epoch', type=int, default=0, help='the number of epochs')
  ap.add_argument('--epochs', type=int, default=11, help='the number of epochs')
  ap.add_argument('--resume', default='', help='the existing model')
  ap.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
  ap.add_argument('--batchsize', type=int, default=10, help='the batch size')
  ap.add_argument('--ngpu', type=int, default=2, help='the number of gpus')
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
  ap.add_argument('--lam_A2B2A', type=float, default=0, help='the weight of cyclic_loss_A2B2A')    
  ap.add_argument('--lam_B2A2B', type=float, default=0.6, help='the weight of cyclic_loss_B2A2B')    
  args=ap.parse_args()

  return args

if __name__ == '__main__':
  opt = parse()
  
  if not os.path.exists(opt.data):
    raise Exception('the directory of training data ' + opt.data + 'doesnot exist !!')
  
  if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)
  print('lam_A2B2A='+str(opt.lam_A2B2A))
  print('lam_B2A2B='+str(opt.lam_B2A2B))
  ## Data preprocessing ##
  # Field for Chinese sentence, and its coresspounding English and CS sentences
  SRC_ZH = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths = True)
  SRC_CS = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths = True)

  # Field for the input of Discriminator A (Chinese sentence: 0, English sentence: 0, CS sentence: 1)
  SRC_DA = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths = True)
  LABEL_DA = Field(sequential=False, use_vocab=False)

  # Field for the input of Discriminator B (Chinese sentence: 1, English sentence: 0, CS sentences: 0)
  SRC_DB = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths = True)
  LABEL_DB = Field(sequential=False, use_vocab=False)

  # Load data to field SRC_ZH, SRC_EN and SRC_CS
  train_data, valid_data, test_data = TabularDataset.splits(
    path=opt.data, train='train_G.json',  validation='valid_G.json', test='test_G.json', format='json',
    fields={"src_zh": ("src_zh", SRC_ZH), "src_cs": ("src_cs", SRC_CS)})

  # Load data to field SRC_DA and LABEL_DA
  train_DA_data, valid_DA_data = TabularDataset.splits(
    path=opt.data, train='train_DA.json',
    validation='valid_DA.json', format='json',
    fields={"text": ("src", SRC_DA), "label": ("label", LABEL_DA)})

  # Load data to field SRC_DB and LABEL_DB
  train_DB_data, valid_DB_data = TabularDataset.splits(
    path=opt.data, train='train_DB.json',
    validation='valid_DB.json', format='json',
    fields={"text": ("src", SRC_DB), "label": ("label", LABEL_DB)})

  # Load data to field SRC_ZH, SRC_EN and SRC_CS
  train_allG_data, valid_allG_data= TabularDataset.splits(
    path=opt.data, train='train_allG_96078.json',  validation='valid_allG_12009.json', format='json',
    fields={"src_zh": ("src_zh", SRC_ZH), "src_cs": ("src_cs", SRC_CS)})

  # Load data to field SRC_ZH, SRC_EN and SRC_CS
  train_PKUSMS_data, valid_PKUSMS_data= TabularDataset.splits(
    path=opt.data, train='Train_smsCorpus_zh.seg.pku.CS.json',  validation='Valid_smsCorpus_zh.seg.pku.CS.json', format='json',
    fields={"src_zh": ("src_zh", SRC_ZH), "src_cs": ("src_cs", SRC_CS)})

  # Build vocabulary
  SRC_ZH.build_vocab(train_data.src_zh, train_data.src_cs, valid_data.src_zh, valid_data.src_cs, test_data.src_zh, test_data.src_cs, train_DA_data.src, valid_DA_data.src, train_DB_data.src, valid_DB_data.src)
  #SRC_EN.vocab = SRC_ZH
  SRC_CS.vocab = SRC_ZH.vocab
  SRC_DA.vocab = SRC_ZH.vocab
  SRC_DB.vocab = SRC_ZH.vocab
  ## [DEBUG] remove it later
  print('SRC_ZH.vocab (SRC_CS.vocab) size '+str(len(SRC_ZH.vocab)))
  print('SRC_DA.vocab size '+str(len(SRC_DA.vocab)))
  print('SRC_DB.vocab size '+str(len(SRC_DB.vocab)))
  ## [DEBUG] remove it later
  
  G_INPUT_DIM = len(SRC_ZH.vocab)
  G_OUTPUT_DIM = len(SRC_DA.vocab)
  DA_INPUT_DIM = len(SRC_DA.vocab)
  DA_OUTPUT_DIM = 2 # 0 is fake_data and 1 is real_data
  DB_INPUT_DIM = len(SRC_DB.vocab)
  DB_OUTPUT_DIM = 2 # 0 is fake_data and 1 is real_data

  G_PAD_IDX = SRC_ZH.vocab.stoi['<pad>']
  G_SOS_IDX = SRC_ZH.vocab.stoi['<sos>']
  G_EOS_IDX = SRC_ZH.vocab.stoi['<eos>']
  DA_PAD_IDX = SRC_DA.vocab.stoi['<pad>']
  DA_SOS_IDX = SRC_DA.vocab.stoi['<sos>']
  DA_EOS_IDX = SRC_DA.vocab.stoi['<eos>']
  DB_PAD_IDX = SRC_DB.vocab.stoi['<pad>']
  DB_SOS_IDX = SRC_DB.vocab.stoi['<sos>']
  DB_EOS_IDX = SRC_DB.vocab.stoi['<eos>']

  ## Using GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  ## Networks setting ##
  # models for Generator
  attn_A2B = Attention(opt.eunits, opt.dunits)
  ENC_A2B = Encoder(G_INPUT_DIM, opt.enc_emb, opt.eunits, opt.dunits, opt.enc_dropout)
  DEC_A2B = Decoder(G_OUTPUT_DIM, opt.dec_emb, opt.eunits, opt.dunits, opt.dec_dropout, attn_A2B)
  netG_A2B = Seq2Seq(ENC_A2B, DEC_A2B, G_PAD_IDX, G_SOS_IDX, G_EOS_IDX, device).to(device)
  netG_A2B.apply(init_weights)

  attn_B2A = Attention(opt.eunits, opt.dunits)
  ENC_B2A = Encoder(G_INPUT_DIM, opt.enc_emb, opt.eunits, opt.dunits, opt.enc_dropout)
  DEC_B2A = Decoder(G_OUTPUT_DIM, opt.dec_emb, opt.eunits, opt.dunits, opt.dec_dropout, attn_B2A)
  netG_B2A = Seq2Seq(ENC_B2A, DEC_B2A, G_PAD_IDX, G_SOS_IDX, G_EOS_IDX, device).to(device)
  netG_B2A.apply(init_weights)
  
  # models for D_A and D_B
  netD_A = Discriminator(DA_INPUT_DIM, DA_OUTPUT_DIM, opt.batchsize, opt.dis_units, opt.dis_emb).to(device)
  netD_B = Discriminator(DB_INPUT_DIM, DB_OUTPUT_DIM, opt.batchsize, opt.dis_units, opt.dis_emb).to(device)
  netD_A.apply(init_weights)
  netD_B.apply(init_weights)
  netD_A.load_state_dict(torch.load('exp/CCG_40K_trylambda/0_bs64_netD_A.pt'))
  netD_B.load_state_dict(torch.load('exp/CCG_40K_trylambda/0_bs64_netD_A.pt'))

  number1=count_parameters(netG_A2B)
  number2=count_parameters(netG_B2A)
  number3=count_parameters(netD_A)
  number4=count_parameters(netD_B)
  print(f'The model has {number1} trainable parameters')
  print(f'The model has {number2} trainable parameters')
  print(f'The model has {number3} trainable parameters')
  print(f'The model has {number4} trainable parameters')
  
  model = CCG(netG_A2B, netG_B2A, netD_A, netD_B, G_PAD_IDX)
  #model.netG_A2B.load_state_dict(torch.load('exp/AISSMS_40K_2/9_bs64_netA2B.pt'))
  #model.netG_B2A.load_state_dict(torch.load('exp/AISSMS_40K_2/9_bs64_netB2A.pt'))
  #model.to(device)
  ## resume ##
  if opt.resume:
      print('resume model '+str(opt.resume))
      model.netG_A2B.load_state_dict(torch.load(opt.resume))
      model.netG_A2B.load_state_dict(torch.load(opt.resume.replace('netA2B.pt','netB2A.pt')))
  else:
      model.netG_A2B.load_state_dict(torch.load('exp/AISSMS_40K_2/9_bs64_netA2B.pt'))
      model.netG_B2A.load_state_dict(torch.load('exp/AISSMS_40K_2/9_bs64_netB2A.pt'))
  model.to(device)

  # Optimizers
  optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

  best_valid_loss = float('inf')

  # Data iterator for Generator (A->B)
  train_A2B_iterator, valid_A2B_iterator = BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = 20,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src_zh),
    device = device)
  
  # Data iterator for Generator (B->A)
  train_B2A_iterator, valid_B2A_iterator = BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = 20,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src_cs),
    device = device)
  
  #for epoch in range(opt.epochs):
  for epoch in range(opt.start_epoch, opt.epochs):
    start_time = time.time()
    
    train_loss = train(model, netD_A, netD_B, train_A2B_iterator, train_B2A_iterator, optimizer, opt.clip, opt.lam_A2B2A, opt.lam_B2A2B, G_PAD_IDX)
    valid_loss = evaluate(model, netD_A, netD_B, valid_A2B_iterator, valid_B2A_iterator , opt.lam_A2B2A, opt.lam_B2A2B)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
    filename = opt.exp_dir + '/CCG'+str(epoch)+'_bs' + str(opt.batchsize) + '_netA2B.pt'
    torch.save(netG_A2B.state_dict(), filename)
    filename = opt.exp_dir + '/CCG'+str(epoch)+'_bs' + str(opt.batchsize) + '_netB2A.pt'
    torch.save(netG_B2A.state_dict(), filename)

    print(f'[CCG] Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t[CCG] Train Loss: {train_loss:.3f}')
    print(f'\t[CCG] Val. Loss: {valid_loss:.3f}')

    if (epoch == 10):
      tag = opt.exp_dir + '/CCG_EP' + str(epoch) + '_bs' + str(opt.batchsize)
      filename = tag + '_train_allG'
      generate_prediction(netG_A2B, train_allG_data, filename)
      filename = tag + '_valid_allG'
      generate_prediction(netG_A2B, valid_allG_data, filename)
      filename = tag + '_test_G'
      generate_prediction(netG_A2B, test_data, filename)
      filename = tag + '_train_PKUSMS'
      generate_prediction(netG_A2B, train_PKUSMS_data, filename)
      filename = tag + '_valid_PKUSMS'
      generate_prediction(netG_A2B, valid_PKUSMS_data, filename)

