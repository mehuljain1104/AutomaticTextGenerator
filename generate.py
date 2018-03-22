import sys
import numpy 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "shaky.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars- seq_length, 1):
	seq_in = raw_text[i:i+seq_length]	
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])


X = numpy.reshape(dataX,(len(dataX),seq_length,1))
X = X/float(n_vocab)
print X.shape

y = np_utils.to_categorical(dataY)
print y.shape

model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

model_name = "weights-improvement-50-1.1264.hdf5"
model.load_weights(model_name)
model.compile(loss='categorical_crossentropy',optimizer='adam')

Start = numpy.random.randint(0,len(dataX)-1)
pattern = dataX[Start]

print "Seed: "
print "\"",''.join([int_to_char[value] for value in pattern]),"\""

for i in range(1000):
	x = numpy.reshape(pattern,(1,len(pattern),1))
	x = x / float(n_vocab)
	prediciton = model.predict(x, verbose=0)
	index = numpy.argmax(prediciton)
	result = int_to_char[index]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print "\nDone"

