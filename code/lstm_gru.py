import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

vocab_size = 100
pad_id = 0

data = [
  [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
  [62,76,79,66,32],
  [93,77,16,67,46,74,24,70],
  [19,83,88,22,57,40,75,82,4,46],
  [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
  [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
  [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
  [94,21,79,24,3,86],
  [80,80,33,63,34,63],
  [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
]

# max_len 구하기, padding하기
max_len = len(max(data, key=len))
valid_lens = []

for i, sentence in enumerate(data):
    valid_lens.append(len(sentence))
    if len(sentence) < max_len:
        data[i] += [pad_id]*(max_len - len(sentence))

# 텐서만들기, 문장 길이 따라 sorting
batch = torch.LongTensor(data)
batch_lens = torch.LongTensor(valid_lens)

sorted_lens, idx = batch_lens.sort(descending = True)
batch = batch[idx]

# LSTM 초기화 & h_0, c_0 초기화
embedding_size = 256
hidden_size = 512
num_layers = 1
num_dirs = 1  #단방향

embedding = nn.Embedding(vocab_size, embedding_size)
lstm = nn.LSTM(
    input_size = embedding_size, 
    hidden_size= hidden_size, 
    num_layers = num_layers
)

h_0 = torch.zeros((num_layers*num_dirs, batch.shape[0], hidden_size))
c_0 = torch.zeros((num_layers*num_dirs, batch.shape[0], hidden_size))

# h_0 = D∗num_layers,N,Hout
# c_0 = D∗num_layers,N,Hcell
# h_0 -> proj_size ??

# # embedding, pack_padded_sequence 적용, LSTM 통과, pad_packed_sequence 적용
batch_emb = embedding(batch)
# print(batch_emb.shape)  # torch.Size([10, 20, 256]) -> batchfirst=False라서 (seq, batch, dim)여야 함

packed_batch = pack_padded_sequence(batch_emb.transpose(0,1), sorted_lens)  # data(모든 단어), lens, _, _
print(packed_batch[0].shape)  #torch.Size([123, 256]) (모든length 합,embedding_size)
print(packed_batch[1])

# h_0,c_0 생략 실험
# output, (h_n, c_n) = lstm(packed_batch, (h_0, c_0))
output, (h_n, c_n) = lstm(packed_batch) # 실험 성공

padded_batch, padded_batch_lens = pad_packed_sequence(output)
print(70, padded_batch.shape) #torch.Size([20, 10, 512])
print(padded_batch_lens)

# == GRU로 task 수행 =============================================================================
# GRU 생성, Linear, "input_id"(첫단어 모음), hidden 초기화
gru = nn.GRU(
    input_size = embedding_size, 
    hidden_size = hidden_size, 
    num_layers= num_layers
)
linear = nn.Linear(hidden_size, vocab_size)

input_id = batch.transpose(0, 1)[0, :]
print(72, input_id)
hidden_0 = torch.zeros((num_layers*num_dirs, batch.shape[0], hidden_size))

# Teacher forcing 없이 이전에 얻은 결과를 다음 input으로 이용
for time in range(max_len):
    input_id = embedding(input_id.unsqueeze(0))  # (10,) ->(1, 10) -> (1, 10, emb_dim)
    output, h_n = gru(input_id, hidden_0)  # seq_len, batch_size, input_size여야 함

    output = linear(output)  # (1, 10, vocab_size)
    output = output.squeeze() # torch.Size([10, 100])
    pred = torch.argmax(output,dim=1)
    input_id = pred
    print(time, pred)
    
# == max vs. argmax ==========================
a = torch.tensor([[0, 1, -1], [1, 4, 2]])
print(100, "#####", torch.max(a, dim=-1))
print(torch.argmax(a, dim=1))
# ============================================

# 양방향, GRU 생성
num_layers = 2
num_dirs = 2
dropout=0.1

gru = nn.GRU(
    input_size = embedding_size, 
    hidden_size = hidden_size, 
    num_layers= num_layers, 
    dropout = dropout, 
    bidirectional=True if num_dirs > 1 else False
)

# 임베딩, pack_padded_sequence, h_0 초기화, gru 통과, pad_packed_sequence
hidden_0 = torch.zeros((num_layers*num_dirs, batch.shape[0], hidden_size))

batch_emb = embedding(batch)

packed_batch = pack_padded_sequence(batch_emb.transpose(0,1), sorted_lens)

output, h_n = gru(packed_batch, hidden_0)

padded_batch, padded_batch_lens = pad_packed_sequence(output)

print(padded_batch.shape, padded_batch_lens.shape)
#torch.Size([20, 10, 1024])
