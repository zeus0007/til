import pprint
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


vocab_size = 100  # 단어 idx 1-100
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

# 1
# 가장 긴 문장에 맞춰서 padding 처리
# "max_len" "valid_lens"
max_len = len(max(data, key=len))#20
valid_data = []  # padding 처리될
data_lens = []
for sentence in data:
    valid_data.append(sentence.copy())
    data_lens.append(len(sentence))
    if len(sentence) < max_len:
        valid_data[-1].extend([pad_id]*(max_len - len(sentence)))
pprint.pprint(valid_data)
pprint.pprint(data)
        

# 2
# tensor로 변환 (batch 하나 통째로)
# "batch"
batch = torch.LongTensor(valid_data)
batch_lens = torch.LongTensor(data_lens)

# 3
# one-hot -> cont vector embedding
# 100차원 -> 256차원 "embedding_size" "embedding"
embedding_size = 256
embedding = nn.Embedding(vocab_size, embedding_size)
embedded_batch = embedding(batch)
print(embedded_batch.shape)

# 4
hidden_size = 512  # RNN의 hidden size
num_layers = 1  # 쌓을 RNN layer의 개수
num_dirs = 1  # 1: 단방향 RNN, 2: 양방향 RNN / bidirectional: (h1+h2 -> h )
# rnn 모델 초기화 & "h_0" 초기화
rnn_model = nn.RNN(
  input_size = embedding_size,
  hidden_size = hidden_size,
  num_layers = num_layers,
  batch_first=True)

h_0 = torch.zeros(num_layers*num_dirs, batch.shape[0], hidden_size)

# 5
# rnn forward 결과
# embedded_batch = (10,20,256)
# (sequence length, batch size, input_size)  batch_first=True
# hidden_state 가 h_n을 포함??
hidden_state, h_n = rnn_model(embedded_batch, h_0) # output == hidden_state
print("here", hidden_state.shape)  # torch.Size([10, 20, 512]) -> 20개 위치에 대한 20개 h
print("here", h_n.shape)  # # torch.Size([1, 10, 512]) ->  마지막 위치에 대한 마지막 1개 h


# 마지막 h_n으로 text classification 수행 (num_classes = 2)
#h_n -> linear -> y_0(num_clases = 2)
num_classes= 2
linear = nn.Linear(hidden_size,num_classes)  # W_hy
y_tc = linear(h_n)
print(y_tc.shape)  # torch.Size([1, 10, 2])

# 7
# "hidden_states"로 token-level의 task를 수행
num_classes = 5
linear = nn.Linear(hidden_size,num_classes)
y_tl = linear(hidden_state)
print(y_tl.shape)  # torch.Size([10, 20, 5])

# 8
# PackedSequence
# 데이터를 padding전 원래 길이 기준으로 정렬합니다. sorted_lens sorted_idx sorted_batch

sorted_lens, sorted_idx = batch_lens.sort(descending=True)
print(sorted_lens, sorted_idx)
sorted_batch = batch[sorted_idx]

# 9
# embedding & pack_padded_sequence() 수행
packed_embedded_batch = embedding(sorted_batch)
packed_input = pack_padded_sequence(packed_embedded_batch, sorted_lens, batch_first=True)
# print(packed_input[0])  # 모든 단어를 합친 하나의 객체 (모든 단어 합친 개수, 차원)
# print(packed_input[1])  # 각 단어 개수를 담은 리스트
# data, batch_sizes, sorted_indices, unsorted_indices

# 10
# rnn에 통과
hidden_state, h_n = rnn_model(packed_input, h_0)

# 기존 rnn에 비해 packed sequence를 사용하며 달라진 점
# 기존 : input-> hidden_state -> y
# now : input-> packed -> hidden -> unpacked -> y

# 11
# pad_packed_sequence() 수행 -> pad 복원 mojo
seq_unpacked, lens_unpacked = pad_packed_sequence(hidden_state , batch_first=True)
print(seq_unpacked.shape)  # torch.Size([10, 20, 512])
print(seq_unpacked.transpose(0, 1))
print(lens_unpacked)  # tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])

# 4시간 만에 rnn 클리어!
# 너무 좋습니다