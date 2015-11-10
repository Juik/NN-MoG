from kmeans import *
import sys
from nn import *
from util import *
import matplotlib.pyplot as plt
plt.ion()

#this function is for 3.2 to find out the best randConst
def mogEM_2(x, K, iters, minVary, rc):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape
  best_rand = 0
  highest_log = -100000
  # Initialize the parameters
  # 1? 5?
  for randConst in rc:
    p = randConst + np.random.rand(K, 1)
    # print p
    p = p / np.sum(p)
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)
   
    # Change the initializaiton with Kmeans here
    #--------------------  Add your code here --------------------  
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
    # print 'initialize with kmeans and iter=5'
    # mu = KMeans(x, K, 5)
    
    #------------------------------------------------------------  
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logProbX = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
      # Do the E step
      respTot = np.zeros((K, 1))
      respX = np.zeros((N, K))
      respDist = np.zeros((N, K))
      logProb = np.zeros((1, T))
      ivary = 1 / vary
      logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
      logPcAndx = np.zeros((K, T))
      for k in xrange(K):
        dis = (x - mu[:,k].reshape(-1, 1))**2
        logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
      
      mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
      mx = np.max(logPcAndx, axis=0).reshape(1, -1)
      PcAndx = np.exp(logPcAndx - mx)
      Px = np.sum(PcAndx, axis=0).reshape(1, -1)
      PcGivenx = PcAndx / Px

      # print '---------------'
      # print PcAndx
      # print '---------------'
      logProb = np.log(Px) + mx
      logProbX[i] = np.sum(logProb)

      # print 'Iter %d logProb %.5f' % (i, logProbX[i])
      if(highest_log < np.sum(logProb)):
        highest_log = np.sum(logProb)
        best_rand = randConst

      # Plot log prob of data
      plt.figure(1);
      plt.clf()
      plt.plot(np.arange(i), logProbX[:i], 'r-')
      plt.title('Log-probability of data versus # iterations of EM')
      plt.xlabel('Iterations of EM')
      plt.ylabel('log P(D)');
      plt.draw()

      respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
      respX = np.zeros((N, K))
      respDist = np.zeros((N,K))
      for k in xrange(K):
        respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
        respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

      # Do the M step
      p = respTot
      mu = respX / respTot.T
      vary = respDist / respTot.T
      vary = (vary >= minVary) * vary + (vary < minVary) * minVary
    
  return highest_log, best_rand




def mogEM(x, K, iters, minVary):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  # 1? 5?
  randConst = 2
  print 'randConst equals : %d' % randConst
  p = randConst + np.random.rand(K, 1)
  # print p
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  # print 'using traditional initialization'
  # mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  print 'initialize with kmeans and iter=5'
  mu = KMeans(x, K, 5)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px

    # print '---------------'
    # print PcAndx
    # print '---------------'
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

# this function is to find the best randConst within {1,2,5,10}
def q2_1():
  iters = 30
  minVary = 0.01
  K = 2
  # inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  inputs_2, inputs_valid_2, inputs_test_2, target_train_2 , target_valid_2, target_test_2 = LoadData('./digits.npz', load3=False)
  inputs_3, inputs_valid_3, inputs_test_3, target_train_3 , target_valid_3, target_test_3 = LoadData('./digits.npz', load2=False)
  rc = {1,2,5,10}
  re_1 = 0
  re_2 = 0
  re_5 = 0
  re_10 = 0
  for i in range(100):
    highest_log, best_rand = mogEM_2(inputs_2, K, iters, minVary, rc)
    print highest_log
    print best_rand
    if best_rand == 1:
      re_1 += 1
    elif best_rand == 2:
      re_2 += 1
    elif best_rand == 5:
      re_5 += 1
    elif best_rand == 10:
      re_10 += 1
  print ' final result is : \n'
  print "1: %d" % re_1
  print "2: %d" % re_2
  print "5: %d" % re_5
  print "10: %d" % re_10

#this is to answer 3.2
def q2():
  iters = 20
  minVary = 0.01
  K = 2
  # inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  inputs_2, inputs_valid_2, inputs_test_2, target_train_2 , target_valid_2, target_test_2 = LoadData('./digits.npz', load3=False)
  inputs_3, inputs_valid_3, inputs_test_3, target_train_3 , target_valid_3, target_test_3 = LoadData('./digits.npz', load2=False)
  
  p2,mu2,vary2,lp2 = mogEM(inputs_2, K, iters, minVary)
  p3,mu3,vary3,lp3 = mogEM(inputs_3, K, iters, minVary)
  raw_input('Press Enter to continue.')

  ShowMeans(mu2)
  ShowMeans(mu3)
  ShowMeans(vary2)
  ShowMeans(vary3)

  print p2
  print p3

  raw_input('Press Enter to continue.')


def q3():
  iters = 20
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # print inputs_train
  K=20
  p, mu, vary, logProbX = mogEM(inputs_train, K, iters, minVary)
  # print p
  # print mu
  # print vary
  # print logProbX
  train_logprob = mogLogProb(p, mu, vary, inputs_train)
  valid_logprob = mogLogProb(p, mu, vary, inputs_valid)
  test_logprob = mogLogProb(p, mu, vary, inputs_test)
  print 'Logprob : Train  %f Valid %f Test %f' % (np.mean(train_logprob), np.mean(valid_logprob), np.mean(test_logprob))
  #print 'mix por:'
  #print p
  # ShowMeans(mu)
  # ShowMeans(vary)

  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  raw_input('Press Enter to continue.')

def q4():
  iters = 10
  minVary = 0.01
  numComponents = np.array([2, 5, 10, 15,20, 25, 30])
  errorTrain = np.zeros(len(numComponents))
  errorTest = np.zeros(len(numComponents))
  errorValidation = np.zeros(len(numComponents))
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, logprobx2 = mogEM(train2, K, iters, minVary)
    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, logprobx3 = mogEM(train3, K, iters, minVary)
    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    errorTrain[t] = get_error(p2,mu2,vary2,p3,mu3,vary3,inputs_train,target_train)
    errorValidation[t] = get_error(p2,mu2,vary2,p3,mu3,vary3,inputs_valid,target_valid)
    errorTest[t] = get_error(p2,mu2,vary2,p3,mu3,vary3,inputs_test,target_test)
    
  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
  plt.title('error rate for different # of clusters')
  plt.plot(numComponents, errorTrain, marker='x',label='Training')
  plt.plot(numComponents, errorValidation, marker='o',label='Validation')
  plt.plot(numComponents, errorTest, marker='v',label='Test')
  plt.ylabel('Classification Error')
  plt.xlabel('mixture components')
  plt.legend()
  plt.grid()
  plt.draw()
  raw_input('Press Enter to continue.')


def get_error(p2, mu2, vary2, p3, mu3, vary3, x, targets):
  lp2 = mogLogProb(p2, mu2, vary2, x)
  # print 'lp2'
  # print lp2.shape
  # print lp2
  lp3 = mogLogProb(p3, mu3, vary3, x)
  # print 'lp3'
  # print lp3.shape
  # print lp3
  # print 'targets'
  # print targets.T
  # print targets.shape
  
  count =0 
  for l2,l3,t in zip(lp2, lp3,targets.T):
    print l2,l3,t

    if (l3>l2 and t==1.) or (l2>l3 and t==0.):
      count +=1
    '''
    if l2>l3 and t==0:
      count += 1
    elif l3>l2 and t==1:
      count += 1
    '''
  # print count
  # print len(lp2)

  return 1 - float(count) / float(len(lp2))

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------
  num_hiddens = 30
  eps = 0.2
  momentum = 0.0
  num_epochs = 2000
  W1, W2, b1, b2, train_error, valid_error, mean_classification_error, train_rate, valid_rate = TrainNN(num_hiddens, eps, momentum, num_epochs)
  print W1
  print '------------'
  print W1[:,:5]
  print '------------'
  print W1[:,:-5]
  print '------------'
  print W1.shape
  print W1[:,:5].shape
  print W1[:,:-5].shape
  ShowMeans(W1[:,:5])
  ShowMeans(W1[:,-5:])
  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  # q2_1()
  # q2()
  q3()
  # q4()
  # q5()

