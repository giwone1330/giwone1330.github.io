---
title: "Building an LSTM Character Generator From Scratch"
description: "A character-level LSTM implementation that connects recurrent modeling theory with gated sequence generation in code."
pubDate: "May 18 2026"
heroImage: "/blog/eee4423/lstm_rnn/GiwonShin_Lab11_files/GiwonShin_Lab11_24_2.png"
badge: "EEE4423"
tags: ["EEE4423", "LSTM", "RNN", "Sequence Modeling", "PyTorch"]
---

## Paper Context

The background section for this project was more theory-heavy than most of the other papers in the series. Instead of only summarizing an application paper, it traced recurrent neural networks back to their sequence-processing equations and then explained why standard RNNs struggle to preserve long-range dependencies during training.

That background also makes the implementation easier to read. Once the cell state is treated as the long-range memory path, the role of the forget, input, and output gates becomes much clearer in the later text-generation results.

## Implementation Walkthrough

The sections below move from preprocessing to the custom LSTM blocks and finally to the generated text samples.

Character-level sequence models like this are a much smaller version of the same recurrent machinery used in translation and captioning.

### Prepare data

The dataset source here is a plain text file. Potential Unicode characters are converted to plain ASCII with the `unidecode` package.


```python
file = unidecode.unidecode(open('lose_yourself_eminem.txt').read())
file_len = len(file)
print('file_len =', file_len)

```
The full text stream is split into chunks before tensorization.

```python
chunk_len = 200

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

print(random_chunk())
```
Each chunk will be turned into a tensor by looping through the characters of the string and looking up the index of each character in `all_characters`.

```python
# Turn string into list of longs
all_characters = string.printable
print(all_characters)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

print('abcDEF is changed to ', char_tensor('abcDEF'))
```
The training pair is assembled from a random chunk. The input contains all characters *up to the last*, and the target contains all characters *from the first*. For a chunk such as "abc", the input corresponds to "ab" while the target corresponds to "bc".

```python
def random_training_set():
    chunk = random_chunk()
    inputs = char_tensor(chunk[:-1])
    targets = char_tensor(chunk[1:])
    return inputs, targets
```

### LSTM model

#### [Diagram of LSTM]
An LSTM consists of a cell state, a hidden state, and three gates that modify or use the cell state. The cell state is the central pathway through which information flows, and the roles of the three gates are summarized below.

#### [Forget Gate]
The forget gate determines which information in the cell state should be erased.

#### [Input Gate]
First, the candidate cell state is created using the current input and the previous hidden state. And the input gate determines how much the candidate cell state is reflected to the cell state.

#### [Output Gate]
The output gate determines which elements should be extracted from the cell state to produce the output.

The above expression is summarized as follows,

This model will take as input the character for step $t_{-1}$ and is expected to output the next character $t$. There are three layers - one linear layer that encodes the input character into an internal state, one LSTM layer that operates on that internal state and a hidden state, and a decoder layer that outputs the probability distribution.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.batch_size = 1

        self.encoder = nn.Embedding(input_dim, hidden_dim)

        # lstm
        # The size of input is (batch_size, seq_dim, hidden_dim)
        # like the previous report, make a private class to use.
        class LSTMCell(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(LSTMCell, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size

                # W term
                self.W_ih = nn.Parameter(
                    torch.Tensor(input_size, 4 * hidden_size))
                self.W_hh = nn.Parameter(
                    torch.Tensor(hidden_size, 4 * hidden_size))

                # B term
                self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
                self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
                self.reset_parameters()

            def reset_parameters(self):
                std = 1.0 / float(self.hidden_size)
                for weight in self.parameters():
                    nn.init.uniform_(weight, -std, std)

            def forward(self, input, hx):
                h_prev, c_prev = hx

                # implementation of the above formula
                gates = torch.matmul(input, self.W_ih) + self.b_ih
                gates += torch.matmul(h_prev, self.W_hh) + self.b_hh

                # splitting the tensor into four subparts
                i_gate, f_gate, o_gate, g_gate = torch.split(
                    gates, self.hidden_size, dim=1)
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                g_gate = torch.tanh(g_gate)

                c = f_gate * c_prev + i_gate * g_gate
                h = o_gate * torch.tanh(c)

                return h, c

        self.layers = nn.ModuleList()
        # First layer input size = hidden_dim (encoder output size)
        self.layers.append(LSTMCell(self.hidden_dim, self.hidden_dim))
        for _ in range(1, self.layer_dim):
            self.layers.append(LSTMCell(self.hidden_dim, self.hidden_dim))

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hn, cn):
        # encode: input is (batch_size,) integer indices -> (batch_size, hidden_dim)
        encoded = self.encoder(input)

        # get the initial value if None
        if hn is None or cn is None:
            hn, cn = self.init_hidden()

        new_h, new_c = [], []
        x = encoded  # (batch_size, hidden_dim)

        # iterate through the layers, indexing hidden state per layer
        for i, layer in enumerate(self.layers):
            h_prev = hn[i]   # (batch_size, hidden_dim)
            c_prev = cn[i]   # (batch_size, hidden_dim)
            h_new, c_new = layer(x, (h_prev, c_prev))
            new_h.append(h_new)
            new_c.append(c_new)
            x = h_new  # output of this layer is input to the next

        hn = torch.stack(new_h, dim=0)  # (layer_dim, batch_size, hidden_dim)
        cn = torch.stack(new_c, dim=0)

        # decode using the last layer's output
        output = self.decoder(x)  # (batch_size, output_dim)
        return output, hn, cn

    def init_hidden(self):
        # The size of h0, c0 should be (layer_dim, batch_size, hidden_dim)
        device = next(self.parameters()).device
        h0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).to(device)
        return h0, c0

hidden_dim = 100
n_layers = 1
n_characters = len(all_characters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(n_characters, hidden_dim, n_layers, n_characters)
model.to(device)

```

 LSTMModel(
 (encoder): Embedding(100, 100)
 (layers): ModuleList(
 (0): LSTMCell()
 )
 (decoder): Linear(in_features=100, out_features=100, bias=True)
 )

### Loss function and optimizer

```python
criterion = nn.CrossEntropyLoss()

lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

### Character-level generation

- Generate a sentence with a length of $predict\_len$, starting from a single character $prime\_str$.
- Example) evaluate(prime_str='Y', predict_len=20) -> You better let it go

```python
def evaluate(prime_str='W', predict_len=100):
    # suppose prime_str is a single character
    # and use greedy search to predict the next character

    model.eval()
    model.batch_size = 1
    hn, cn = model.init_hidden()
    predicted = prime_str

    prime_tensor = char_tensor(prime_str).to(device)  # shape (1,)

    for i in range(predict_len):
        with torch.no_grad():
            output, hn, cn = model(prime_tensor, hn, cn)
            _, topi = output.topk(1)
            predicted_char = all_characters[topi.item()]

            predicted += predicted_char
            prime_tensor = torch.tensor(
                [topi.item()], dtype=torch.long).to(device)  # shape (1,)

    return predicted

```

### Training loop

- Plot the training loss curve.
- Print the output sentence with a length of 100, using $evaluate()$ function.

```python
n_epochs = 2000
print_every = 100
plot_every = 10

all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    model.train()
    model.batch_size = 1

    # Load text
    inputs, targets = random_training_set()
    if inputs.size()[0] < 200:  # always pick full 200 length set
        continue

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass: iterate through the sequence one character at a time
    loss = 0
    inputs = inputs.to(device)    # (chunk_len,) integer indices
    targets = targets.to(device)  # (chunk_len,) integer indices

    h, c = model.init_hidden()
    for t in range(chunk_len):
        output, h, c = model(inputs[t:t+1], h, c)   # output: (1, n_characters)
        loss += criterion(output, targets[t:t+1])    # targets[t:t+1]: (1,)

    # Backward pass
    loss.backward()

    # Updating parameters
    optimizer.step()
    loss_avg += loss.item() / chunk_len

    if epoch % print_every == 0:
        print('*'*25, 'epoch%d'%epoch, '*'*25)
        print('loss %.4f'%loss.item())
        print(evaluate('I', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure()
plt.plot(all_losses)

```
![png](/blog/eee4423/lstm_rnn/GiwonShin_Lab11_files/GiwonShin_Lab11_24_2.png)

# Discussion

Tried implementation of LSTM without using `nn.LSTM` but failed initially. Coming back to it, the following bugs were identified and fixed.

---

### Fixes Made

1. **File path** — The dataset path pointed to a non-existent directory (`../dataset-dllab/lab11/`). Fixed to use the local file `lose_yourself_eminem.txt`.

2. **`LSTMCell` input size mismatch** — The first `LSTMCell` was constructed with `input_size = input_dim` (vocabulary size), but the encoder (`nn.Embedding`) outputs vectors of size `hidden_dim`. Fixed to `LSTMCell(hidden_dim, hidden_dim)`.

3. **`forward()`: wrong `None` check for hidden state** — The original code assigned `self.init_hidden()` (which returns a tuple `(h0, c0)`) to `hn` alone, leaving `cn` unset. Fixed to `hn, cn = self.init_hidden()`.

4. **`forward()`: hidden state not indexed per layer** — `h_prev` and `c_prev` were set to the full `(layer_dim, batch, hidden)` tensors instead of being sliced per layer. Fixed to `h_prev = hn[i]` inside the layer loop.

5. **`forward()`: layer output not threaded through** — The encoded input `x` was passed to every layer instead of being updated with each layer's output. Fixed by adding `x = h_new` after each layer.

6. **`init_hidden()`: hardcoded `.cuda()`** — This causes a crash on CPU-only machines. Fixed to use `.to(next(self.parameters()).device)` to dynamically target the correct device.

7. **`evaluate()`: re-initialization inside the loop** — `model.eval()` and `model.init_hidden()` were called inside the prediction loop, resetting hidden state every step. Moved outside so the hidden state carries across time steps correctly.

8. **Training loop: broken sequence iteration** — `inputs.view(-1, n_characters)` treated character indices as a batch dimension, completely destroying the sequential structure. Replaced with a character-by-character loop (`for t in range(chunk_len)`), which is the correct BPTT approach.

---

### Analysis

**Early epochs (0–~400):** The model initially learns to copy or guess the first few characters of a sequence, but quickly falls into a repetition loop — repeating either a single token (e.g., `"I the the the"`) or a short phrase (e.g., `"You one shot on the moment You one shot on the moment"`). This is a well-known failure mode of sequence models early in training: the decoder confidently predicts the most frequent token at each step, which feeds back as the next input and creates a fixed point.

**Mid training (~400–800 epochs):** The model begins to break out of single-token repetition and starts cycling through slightly longer repeated chunks. This suggests the hidden state is beginning to encode short-range context, but is not yet stable enough to maintain diversity over longer sequences.

**Later epochs (~800+):** The output becomes noticeably less repetitive. The model starts producing word-like sequences that loosely resemble the training text's style, indicating that the LSTM's cell state is now retaining and leveraging longer-range dependencies.

**Loss curve:** The training loss does decrease overall, showing the model is learning. However, it fluctuates heavily throughout training rather than converging smoothly. This is likely due to: (1) the high learning rate (`lr = 0.005`) combined with a character-by-character forward pass accumulating a large gradient, (2) the lack of gradient clipping, and (3) the stochastic nature of randomly sampled chunks each epoch. Adding gradient clipping (`torch.nn.utils.clip_grad_norm_`) would likely stabilize training.

### *References*
[1] [practical pytorch](https://github.com/spro/practical-pytorch)(https://github.com/spro/practical-pytorch)

[2] [CS 231n](http://cs231n.stanford.edu/syllabus.html)(http://cs231n.stanford.edu/syllabus.html)
