---
title: "Building Seq2Seq With Attention From Scratch"
description: "A machine-translation walkthrough of encoder-decoder sequence modeling, attention, training, and evaluation."
pubDate: "May 18 2026"
heroImage: "/blog/eee4423/seq2seq/GiwonShin_Lab12_files/GiwonShin_Lab12_19_2.png"
badge: "EEE4423"
tags: ["EEE4423", "Seq2Seq", "Attention", "Machine Translation", "PyTorch"]
---

## Paper Context

The background reading tied together two ideas that are easy to separate too cleanly in hindsight: sequence-to-sequence learning and the need for better recurrent units. Encoder-decoder models made variable-length sequence mapping practical, but compressing an entire source sentence into a single fixed vector created a bottleneck, especially for longer translations.

Attention is the part that makes the model feel workable rather than bottlenecked. Instead of forcing the decoder to depend on one compressed summary, it can revisit encoder states at each output step, which is exactly the behavior that shows up later in the translation examples.

## Implementation Walkthrough

The sections below cover the preprocessing pipeline, the encoder-decoder with attention, and a few translation examples from the trained model.

### Prepare data

The dataset for this project is a collection of many thousands of English-to-French translation pairs. In this implementation it comes from <https://download.pytorch.org/tutorial/data.zip>, where the file is stored as a tab-separated list of translation pairs:

 I am cold. J'ai froid.


```python
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join( c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    # Read the file and split into lines
    lines = open('../dataset-dllab/lab12/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:", input_lang.name, '=', input_lang.n_words, output_lang.name, '=', output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
```
```python
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
```

### Seq2Seq model


[sequence to sequence network](https://arxiv.org/abs/1409.3215) is a model in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a single vector,
and a decoder network unfolds that vector into a new sequence.

Unlike sequence prediction with a single RNN, where every input
corresponds to an output, the seq2seq model frees us from sequence
length and order, which makes it ideal for translation between two
languages.

#### Encoder
The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

#### GRU
The GRU operates using a reset gate (r) and an update gate (z). The candidate state is created by using the previous hidden state and the current input. It is the reset gate that determines how the previous hidden state affects the candidate state. The newly created candidate state and the previous hidden state create a new hidden state, in which the update gate plays a role in balancing the two.

#### LSTM vs GRU

| <center>LSTM</center> | <center>GRU</center> |
|:--------|--------|
| LSTM has 3 gates (forget, input, output) | GRU has 2 gates (reset, update) |
| There is an internal memory (cell state) | There is no cell state and only hidden state exists |
| When making output, another non-linearity is applied | There is no additional non-linearity when making output |

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderRNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # gru
        # The size of input is (batch_size, seq_dim, hidden_dim)
        self.batch_size = 1
        self.layer_dim = 1
        self.input_dim = input_dim
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, input, hn):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hn = self.gru(embedded, hn)
        return output, hn

    def initHidden(self):
        # The size of h0 should be (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim, device=device)
        return h0

hidden_dim = 256
encoder = EncoderRNN(input_lang.n_words, hidden_dim).to(device)
```

#### Decoder
If only the context vector is passed betweeen the encoder and decoder, that single vector carries the burden of encoding the entire sentence. Attention allows the decoder network to "focus" on a specific part of
the encoder's outputs for every step and thus help the decoder choose the right output words.
The attention weights are calculated with a small feed-forward layer that takes the decoder input and hidden state, then applies those weights to the encoder outputs to build a context vector for the current step. Because this attention module is implemented with a fixed-size linear layer, a maximum sentence length has to be chosen ahead of time. Sentences at that limit use the full attention window, while shorter ones use only the relevant prefix.

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        # attention
        # Note that the column of the attention weights is MAX_LENGTH
        # Note that concatenation is used when "attn" and "attn_combine" are created
        self.batch_size = 1
        self.layer_dim = 1
        self.max_length = MAX_LENGTH

        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length) # concatenation to twice the hidden_dim
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # gru
        # The size of input is (batch_size, seq_dim, hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hn, encoder_outputs):
        input = self.embedding(input).view(1, 1, -1)
        input = self.dropout(input)

        # attention
        # All specifications of the operations are described in the above figure (e.g. use ReLU)
        # bmm is a operation which performs a batch matrix-matrix product
        # For simplicity, only 1 layer forward is implemented
        embedded = input[0]
        prev_hidden = hn[0]

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, prev_hidden), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        attn_combine_input = torch.cat((embedded, attn_applied[0]), 1)
        attn_combine_output = self.attn_combine(attn_combine_input).unsqueeze(0)

        atten_input = F.relu(attn_combine_output)

        # gru
        output, hn = self.gru(atten_input, hn)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hn

    def initHidden(self):
        # The size of h0 should be (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim, device=device)
        return h0

decoder = AttnDecoderRNN(hidden_dim, output_lang.n_words, dropout_p=0.1).to(device)
```

### Loss function and optimizer

```python
criterion = nn.NLLLoss()

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
```

### Write the evaluation code

- Using the trained model, display the translated output given input sentence.

```python
def evaluate(sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.initHidden() # initialize hidden state

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_dim, device=device)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoded_words = []

        # Encoder iteration
        for encoder_index in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[encoder_index],
                                                     encoder_hidden)
            encoder_outputs[encoder_index] += encoder_output[0, 0]

        # Context vector delivered to decoder
        decoder_inputs = encoder_outputs
        decoder_hidden = encoder_hidden

        # Decoder iteration
        for decoder_index in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, decoder_inputs)
            topv, topi = decoder_output.data.topk(1)

            # coninue appending generated output until <End of Sentence> token
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach() # detach() to strip off gradients

        return decoded_words

def evaluateRandomly():
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
```

### Training loop

- During training, use the `Teacher forcing` concept in addition to a naive approach.
 - In other words, instead of using the decoder's guess as the next input, the real target outputs are also used sometimes. This shows faster convergence.
- Plot the training loss curve.
- Show the result using $evaluateRandomly()$ function. Below is an example.
*************************************************************
 > il est en train de peindre un tableau . (input)
 = he is painting a picture . (target)
 < he is painting a picture . (output)
*************************************************************

```python
n_iters = 50000
print_every = 1000
plot_every =100

plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

for iter in range(1, n_iters+1):
    # Load data
    training_pair = training_pairs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    # Clear gradients w.r.t. parameters
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Forward pass
    loss = 0
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # initialize encoder hidden state
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_dim, device=device)
    # unroll to input_tensor
    for encoder_index in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[encoder_index], encoder_hidden)
        encoder_outputs[encoder_index] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # Context vector delivered to decoder
    decoder_inputs = encoder_outputs
    decoder_hidden = encoder_hidden

    for decoder_index in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[decoder_index])

        if random.random() > 0.5: # 50:50 chance for teacher forcing and naive approach
            # Teacher forcing
            decoder_input = target_tensor[decoder_index]

        else:
            # Naive approach
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS_token:
                break

    # Backward pass
    loss.backward()

    # Updating parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    print_loss_total += loss.item() / target_length
    plot_loss_total += loss.item() / target_length

    if iter % print_every == 0:
        print('*'*25, 'iter%d'%iter, '*'*25)
        print('loss %.4f'%loss)
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        evaluateRandomly()

    if iter % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure()
plt.plot(plot_losses)
```
![png](/blog/eee4423/seq2seq/GiwonShin_Lab12_files/GiwonShin_Lab12_19_2.png)

# Discussion

After 50,000 iterations, the model was already producing readable short translations, which made the effect of attention fairly easy to see.

## Quantitative analysis
The training of the ENG - FRA translator was completed after 50000 iterations, decreasing the loss from 34.9 to 0.25.

## Qualitative analysis
As training progressed, the outputs moved from broken fragments to short complete sentences, and the translated meaning became noticeably closer to the reference.

---

In the above implementation, I used GRU module from pytorch framework. However, this module can also be made from scratch. Below is the implementaion for custom GRU.

```python
import numpy as np

class myGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(myGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        class GRUCell(nn.Module):
            def __init__(self, input_size, hidden_size, bias=True):
                super(GRUCell, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.bias = bias

                self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
                self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

                self.reset_parameters()

            def reset_parameters(self):
                std = 1.0 / np.sqrt(self.hidden_size)
                for w in self.parameters():
                    w.data.uniform_(-std, std)

            def forward(self, input, hx=None):

                if hx is None:
                    hx = torch.autograd.Variable(input.new_zeros(input.size(0), self.hidden_size))

                x_t = self.x2h(input)
                h_t = self.h2h(hx)

                x_reset, x_upd, x_new = x_t.chunk(3, 1)
                h_reset, h_upd, h_new = h_t.chunk(3, 1)

                reset_gate = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))

                hy = update_gate * hx + (1 - update_gate) * new_gate

                return hy

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))

    def forward(self, input, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.autograd.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = torch.autograd.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l)

        out = outs[-1].unsqueeze(0) # unsqueeze to layer dim
        hx = out

        return out, hx
```

This can be swapped with nn.GRU to produce identical results if needed.

### *References*
[1] [practical pytorch](https://github.com/spro/practical-pytorch)(https://github.com/spro/practical-pytorch)
