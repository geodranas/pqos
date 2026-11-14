import torch.nn as nn
import gparams
from torch.autograd import Variable
import torch
import torch.nn.functional as F
class biLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,bidirectional=True)

        self.fc_1 = nn.Linear(self.num_layers*hidden_size*2, 128)
        self.fc_2 = nn.Linear(128, num_classes)


        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.num_layers*self.hidden_size*2)
        out = self.relu(h_out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out

class batchLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,bidirectional=False)

        self.fc_1 = nn.Linear(self.num_layers*hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)


        self.relu = nn.ReLU()

    def init_states(self,batch_size):
        self.h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        self.c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, x):


        ula, (h_out, _) = self.lstm(x, (self.h_n, self.c_n))
        h_out = h_out.view(-1, self.num_layers*self.hidden_size)
        out = self.relu(h_out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out

class LSTM_bibatch(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM_bibatch, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.bi=gparams._BIDIRECTIONAL
        self.stateful=gparams._STATEFUL
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,bidirectional=self.bi,dropout=gparams._DROPOUT)

        self.fc_small=nn.Linear(self.num_layers*hidden_size*self.multiply_factor, num_classes)
        self.dropout = nn.Dropout(gparams._DROPOUT)
        self.h_n=None
        self.c_n=None

        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')

    def reset_states(self):
        self.h_n=None
        self.c_n=None

    def forward(self, x):

        if (not self.stateful) or (self.h_n is None and self.c_n is None):
            batch=x.shape[0]
            self.h_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))
            self.c_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))

        else:
            pass


        ula, (h_out, c_out) = self.lstm(x, (self.h_n, self.c_n))


        self.h_n=h_out.detach()
        self.c_n =c_out.detach()

        out = h_out.view(-1, self.num_layers*self.hidden_size*self.multiply_factor)

        if self.activation is not None:
            out = self.activation(out)

        out = self.fc_small(out)
        return out

class CNN_batch(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,batch_size):
        super(CNN_batch, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.bi=gparams._BIDIRECTIONAL
        self.stateful=gparams._STATEFUL
        self.linear_big=True
        self.multiply_factor=1
        self.batch=batch_size
        self.dbg=True

        self.cnn_out=128
        self.cnn = nn.Conv1d(self.batch, self.cnn_out, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)

        print(str(self.batch))

        self.fc_1 = nn.Linear(self.cnn_out*self.batch, 1024)
        self.fc_2 = nn.Linear(1024, 64)
        self.fc_3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(gparams._DROPOUT)

        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')


    def reset_states(self):
        pass

    def forward(self, x):
        if self.dbg:
            print('Initial size=' + str(x.size()))
        x = self.cnn(x)
        if self.dbg:
            print('After cnn=' + str(x.size()))
        if self.activation is not None:
            x = self.activation(x)
        x = self.maxpool(x)
        if self.dbg:
            print('After maxpool=' + str(x.size()))

        if self.activation is not None:
            x = self.activation(x)
        if self.dbg:
            print('After activation=' + str(x.size()))

        x = x.view(-1)
        if self.dbg:
            print('After flatten=' + str(x.size()))

        out = self.fc_1(x)
        if self.activation is not None:
            out = self.activation(out)
        out=self.dropout(out)
        out = self.fc_2(out)
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_3(out)
        self.dbg = False
        return out

class LSTM_bibatch_stacked(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM_bibatch_stacked, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.bi=gparams._BIDIRECTIONAL
        self.stateful=gparams._STATEFUL
        self.linear_big=True
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor = 1

        self.hidden1=64
        self.hidden2 = 128
        self.hidden3 = 512
        self.hidden4 = 128
        self.hidden5 = 64

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden1,num_layers=1, batch_first=True,bidirectional=self.bi)
        self.lstm2 = nn.LSTM(input_size=self.hidden1, hidden_size=self.hidden2, num_layers=1, batch_first=True,bidirectional=self.bi)
        self.lstm3 = nn.LSTM(input_size=self.hidden2, hidden_size=self.hidden3,num_layers=1, batch_first=True,bidirectional=self.bi)
        self.lstm4 = nn.LSTM(input_size=self.hidden3, hidden_size=self.hidden4, num_layers=1, batch_first=True,bidirectional=self.bi)
        self.lstm5 = nn.LSTM(input_size=self.hidden4, hidden_size=self.hidden5, num_layers=1, batch_first=True,bidirectional=self.bi)

        self.fc_1 = nn.Linear(self.hidden5, 1024)
        self.fc_2 = nn.Linear(1024, 64)
        self.fc_3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(gparams._DROPOUT)

        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')

    def reset_states(self):
        self.h_n=None
        self.c_n=None

    def forward(self, x):
        batch=x.shape[0]

        h_1 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden1))
        c_1 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden1))
        ula, (h_out, c_out) = self.lstm1(x, (h_1, c_1))
        ula = self.dropout(ula)

        h_2 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden2))
        c_2 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden2))
        ula, (h_out, c_out) = self.lstm2(ula, (h_2, c_2))
        ula = self.dropout(ula)

        h_3 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden3))
        c_3 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden3))
        ula, (h_out, c_out) = self.lstm3(ula, (h_3, c_3))
        ula = self.dropout(ula)

        h_4 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden4))
        c_4 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden4))
        ula, (h_out, c_out) = self.lstm4(ula, (h_4, c_4))
        ula = self.dropout(ula)

        h_5 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden5))
        c_5 = Variable(torch.zeros(1*self.multiply_factor, batch, self.hidden5))
        ula, (h_out, c_out) = self.lstm5(ula, (h_5, c_5))

        out = h_out.view(-1,self.hidden5)

        if self.activation is not None:
            out = self.activation(out)

        out = self.fc_1(out)
        if self.activation is not None:
            out = self.activation(out)
        out=self.dropout(out)
        out = self.fc_2(out)
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_3(out)

        return out

class LSTM_CNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM_CNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.bi=gparams._BIDIRECTIONAL
        self.stateful=gparams._STATEFUL
        self.linear_big=False
        self.debug_sizes=True
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor = 1

        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,bidirectional=self.bi)

        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=5, kernel_size=2)
        self.maxpool = nn.MaxPool1d(kernel_size=1)

        self.fc_1 = nn.Linear(self.num_layers*hidden_size*self.multiply_factor, 1024)
        self.fc_2 = nn.Linear(1024, 64)
        self.fc_3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(gparams._DROPOUT)
        self.h_n=None
        self.c_n=None

        self.fc_small=nn.Linear(self.num_layers*hidden_size*self.multiply_factor, num_classes)

        self.fc_a = nn.Linear(self.num_layers * hidden_size * self.multiply_factor, 1024)
        self.fc_b = nn.Linear(1024, 512)
        self.fc_c = nn.Linear(512, 256)
        self.fc_d = nn.Linear(256, 128)
        self.fc_e = nn.Linear(128, 64)
        self.fc_f = nn.Linear(64, 32)
        self.fc_g = nn.Linear(32, 16)
        self.fc_h = nn.Linear(16, 8)
        self.fc_i = nn.Linear(8, 4)
        self.fc_j = nn.Linear(4, 2)
        self.fc_k = nn.Linear(2, num_classes)

        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')

    def reset_states(self):
        self.h_n=None
        self.c_n=None

    def forward(self, x):
        if self.debug_sizes:
            print('Initial size='+str(x.size()))
        x=self.cnn(x)
        if self.debug_sizes:
            print('After cnn=' + str(x.size()))
        if self.activation is not None:
            x = self.activation(x)
        x=self.maxpool(x)
        if self.debug_sizes:
            print('After maxpool=' + str(x.size()))
        if self.activation is not None:
            x=self.activation(x)
        if self.debug_sizes:
            print('After activation=' + str(x.size()))
            self.debug_sizes=False

        if (not self.stateful) or (self.h_n is None and self.c_n is None):
            batch=x.shape[0]
            self.h_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))
            self.c_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))

        else:
            pass


        ula, (h_out, c_out) = self.lstm(x, (self.h_n, self.c_n))


        self.h_n=h_out.detach()
        self.c_n =c_out.detach()

        out = h_out.view(-1, self.num_layers*self.hidden_size*self.multiply_factor)

        if self.activation is not None:
            out = self.activation(out)

        if self.linear_big:
            out = self.fc_1(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)
            out = self.fc_2(out)
            if self.activation is not None:
                out = self.activation(out)
            out = self.dropout(out)
            out = self.fc_3(out)
        else:
            out = self.fc_a(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_b(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_c(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_d(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_e(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_f(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_g(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_h(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_i(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_j(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)

            out = self.fc_k(out)

        return out
class GRU_bibatch(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GRU_bibatch, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.bi=gparams._BIDIRECTIONAL
        self.linear_big=False
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor = 1
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,bidirectional=self.bi)

        self.fc_1 = nn.Linear(self.num_layers*hidden_size*self.multiply_factor, 2048)
        self.fc_2 = nn.Linear(2048, 32)
        self.fc_3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.25)

        self.fc_small=nn.Linear(self.num_layers*hidden_size*self.multiply_factor, num_classes)

        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')


    def forward(self, x):


        batch=x.shape[0]
        self.h_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))


        ula, h_out = self.gru(x, self.h_n)
        out = h_out.view(-1, self.num_layers*self.hidden_size*self.multiply_factor)

        if self.activation is not None:
            out = self.activation(out)

        if self.linear_big:
            out = self.fc_1(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)
            out = self.fc_2(out)
            if self.activation is not None:
                out = self.activation(out)
            out = self.dropout(out)
            out = self.fc_3(out)
        else:
            out=self.fc_small(out)

        return out

class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi=True
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor=1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2,bidirectional=self.bi)
        self.relu = nn.ReLU()
        self.h_n = None
        self.c_n = None

    def forward(self, x) -> torch.tensor:
        batch=x.shape[0]
        self.h_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))
        self.c_n = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))
        ula, (h_out, c_out) = self.lstm(x, (self.h_n, self.c_n))
        return self.relu(h_out[-1])

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, input_seq_size: int):
        super(DecoderLSTM, self).__init__()
        self.bi=True
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor=1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2,bidirectional=self.bi)
        self.input_seq_size = input_seq_size
        self.h_0=None
        self.c_0 = None
        self.relu = nn.ReLU()
    def forward(self, z: torch.tensor) -> torch.tensor:
        z = z.unsqueeze(1)
        z = z.repeat(1, self.input_seq_size, 1)

        batch=z.shape[0]
        self.h_0 = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))
        self.c_0 = Variable(torch.zeros(self.num_layers*self.multiply_factor, batch, self.hidden_size))

        output, (h_n, c_n) = self.lstm(z, (self.h_0, self.c_0))
        return self.relu(output)

class LSTM_seq2seq(nn.Module):
    def __init__(self, input_size: int, input_seq_size: int, hidden_size: int, num_layers: int, batch_size: int,
                 decoder_output_size: int):

        super(LSTM_seq2seq, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_seq_size = input_seq_size
        self.enc_bi=True
        self.dec_bi=True
        if self.dec_bi:
            self.dec_factor=2
        else:
            self.dec_factor=1
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.encoder = EncoderLSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.decoder = DecoderLSTM(input_size=hidden_size,hidden_size=decoder_output_size,num_layers=num_layers,
                                   input_seq_size=input_seq_size)

        self.fc_1 = nn.Linear(self.input_seq_size*self.dec_factor, decoder_output_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        z = self.encoder(x)
        decoded = self.decoder(z)


        h_out = decoded.reshape(-1,self.input_seq_size*self.dec_factor)


        out = self.fc_1(h_out)


        return out


class LSTM_CNN_old(nn.Module):

    def __init__(self, num_classes, input_size, out_size, num_layers):
        super(LSTM_CNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.out_size = out_size
        self.seq_length = gparams._SLIDING_WINDOW
        self.h_n =None
        self.c_n =None
        self.linear_big=True
        self.bi=gparams._BIDIRECTIONAL
        if self.bi:
            self.multiply_factor=2
        else:
            self.multiply_factor = 1

        self.c1=nn.Conv1d(in_channels=self.input_size, out_channels= self.out_size, kernel_size=1)

        self.fc_1 = nn.Linear(self.num_layers*out_size*self.multiply_factor, 2048)
        self.fc_2 = nn.Linear(2048, 32)
        self.fc_3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.fc_small=nn.Linear(self.num_layers*out_size*self.multiply_factor, num_classes)
        self.pool=nn.MaxPool1d(kernel_size=1)
        if gparams._ACTIVATION == 'relu':
            self.activation=nn.ReLU()
        elif gparams._ACTIVATION=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None
            print('Cannot find activation')

    def forward(self, x):

        x = self.c1(x)
        if self.activation is not None:
            x = self.activation(x)

        out = x.view(-1, self.out_size)

        if self.linear_big:
            out = self.fc_1(out)
            if self.activation is not None:
                out = self.activation(out)
            out=self.dropout(out)
            out = self.fc_2(out)
            if self.activation is not None:
                out = self.activation(out)
            out = self.dropout(out)
            out = self.fc_3(out)
        else:
            out=self.fc_small(out)


        return out


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc3 = nn.Linear(3072, 512)
        self.fc5 = nn.Linear(512, 10)
        self.size=float(6.1)

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

    def get_train_time_mobile_with_epochs(self,samples,epochs):
        return float(epochs*(samples/125))
