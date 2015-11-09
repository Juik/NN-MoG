from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

#this func helps calculate how many "1"s after rounding to decimal 0
def Calculate_nozero(matrix):
  count = 0
  for num in matrix[0]:
    # print num
    if num!=0:
      count +=1
  return count


def TrainNN(num_hiddens, eps, momentum, num_epochs):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  train_rate = []
  valid_rate = []
  mean_classification_error =[]
  num_train_cases = inputs_train.shape[1]
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
    # print prediction

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))
    train_fr = compare_two_lists(np.around(prediction,decimals=0)[0],target_train[0])


    ##############################################
    # rounded= np.around(prediction, decimals=0)
    # print '\nstart'
    # print prediction
    # print '------------'
    # print target_train
    # print '------------'
    # print rounded
    # print '------------'
    # round= target_train - rounded
    # notzero= np.flatnonzero(round)
    # print notzero
    # print len(notzero)
    # MCE =(len(notzero)/600.)
    round = np.around(prediction, decimals = 0)
    # print 'round now begin:'
    # print round
    # print Calculate_nozero(target_train - round)
    MCE = Calculate_nozero(target_train - round)/600.0

    ##############################################



    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    valid_CE = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
    valid_fr = Evaluate_frac(inputs_valid, target_valid, W1, W2, b1, b2)

    train_error.append(train_CE)
    valid_error.append(valid_CE)

    train_rate.append(train_fr)
    valid_rate.append(valid_fr)

    mean_classification_error.append(MCE)

    sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f mean_classification_error %.5F' % (epoch, train_CE, valid_CE,MCE))
    sys.stdout.flush()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  # print '\n-------------'
  # print train_rate[-1]
  # print valid_rate[-1]

  sys.stdout.write('\n')
  final_train_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_error = Evaluate(inputs_test, target_test, W1, W2, b1, b2)

  final_train_rate = Evaluate_frac(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_rate = Evaluate_frac(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_rate = Evaluate_frac(inputs_test, target_test, W1, W2, b1, b2)

  print 'Error: Train %.5f Validation %.5f Test %.5f' % (final_train_error, final_valid_error, final_test_error)
  print 'fr_rate: Train %.5f Validation %.5f Test %.5f' % (final_train_rate, final_valid_rate, final_test_rate)
  return W1, W2, b1, b2, train_error, valid_error, mean_classification_error, train_rate, valid_rate

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  return CE

#this func helps do Evaluate_frac
def compare_two_lists(aa,bb):
  count = 0
  correct_count = 0
  for a in aa:
      if a== bb[count]:
          correct_count = correct_count + 1
      count = count + 1

  return 1 - correct_count / float(count)

#this func helps evaluate fraction rate
def Evaluate_frac(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  fr = compare_two_lists(np.around(prediction,decimals=0)[0],target[0])
  return fr



def DisplayErrorPlot(train_error, valid_error,mean_classification_error):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  plt.plot(range(len(mean_classification_error)), mean_classification_error, 'x', label='Mean of Classification error')
  plt.xlabel('Epochs')
  plt.ylabel('Cross entropy')
  plt.legend()
  plt.grid()
  plt.draw()
  raw_input('Press Enter to exit.')

def DisplayFractionPlot(train_rate,valid_rate):
  plt.figure(2)
  plt.clf()
  plt.plot(range(len(train_rate)),train_rate,'b',label='Train')
  plt.plot(range(len(valid_rate)),valid_rate,'g',label='Valid')
  plt.xlabel('Epochs')
  plt.ylabel('Fraction Rate')
  plt.legend()
  plt.grid()
  plt.draw()
  raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def main():
  num_hiddens = 10
  eps = 0.02
  momentum = 0.5
  num_epochs = 1000
  print num_hiddens
  W1, W2, b1, b2, train_error, valid_error, mean_classification_error, train_rate, valid_rate = TrainNN(num_hiddens, eps, momentum, num_epochs)
  DisplayErrorPlot(train_error, valid_error,mean_classification_error) 
  DisplayFractionPlot(train_rate,valid_rate)

if __name__ == '__main__':
  main()
