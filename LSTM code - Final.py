import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
import glob
import os

path = "C:\\Users\\Owner\\Desktop\\Fall_2017\\F17_MLSP\\Final Project - MLSP\\UCI HAR Dataset\\"
train = path + "train\\Inertial Signals"
test = path + "test\\Inertial Signals"

x_train_path = glob.glob(os.path.join(os.getcwd(), train, "*.txt"))
x_test_path = glob.glob(os.path.join(os.getcwd(), test, "*.txt"))
y_train_path = path + "train\\y_train.txt"
y_test_path = path + "test\\y_test.txt"

def read_inputs(path):
    X = []
    for row in path:
        file = open(row, 'r')
        data_list = file.readlines()
        for i in range(len(data_list)):
            data_list[i] = data_list[i].replace('  ', ' ').strip().split(' ')
        c = np.array(data_list, dtype = np.float32)
        X.append(c)
    X = np.transpose(np.array(X, dtype = np.float32),(1,2,0))
    file.close()
    return(X)
    
def read_outputs(path):
    file = open(path, 'r')
    y = np.array([str1 for str1 in [str2.replace('  ', ' ').strip().split(' ') for str2 in file]], dtype = np.int32)
    y -= 1
    file.close()
    return y

X_train = read_inputs(x_train_path)
X_test = read_inputs(x_test_path)
y_train = read_outputs(y_train_path)
y_test = read_outputs(y_test_path)

n_train, timesteps, input_params = X_train.shape
n_test = X_test.shape[0]

hidden_units = 100
classes = 6

n_iter = 2300000
n_samples = 2000
d_iter = 50000

def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(hidden_units, forget_bias = 0.0, state_is_tuple=True, reuse = tf.get_variable_scope().reuse)

#x1 = x_mod

def LSTM_RNN(x1, W, b):
    z = tf.matmul(x1, W['h_layer']) + b['h_layer']
    activation = tf.nn.relu(z)
    
    x2 = tf.split(activation, timesteps) 
    lstm_1 = lstm_cell() 
    lstm_2 = lstm_cell() 
    lstm_stack = tf.contrib.rnn.MultiRNNCell([lstm_1, lstm_2], state_is_tuple=True)
    
    out, pos = tf.contrib.rnn.static_rnn(lstm_stack, x2, dtype = tf.float32)
    
    lstm_out = out[-1]
    state_est = tf.matmul(lstm_out, W['o_layer']) + b['o_layer']
    
    return state_est


def extract_n_samples(_train, step, n_samples):
    # Taking n samples of data at a time frmom the dataset to generalize better. 
    shape = list(_train.shape)
    shape[0] = n_samples
    batch_s = np.empty(shape)

    for i in range(n_samples):
        index = ((step-1)*n_samples + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s

def y_wrap_fn(y):
    y = y.reshape(len(y))
    n_values = int(np.max(y)) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]

# Input and Ouput tensors
x = tf.placeholder(tf.float32, shape = (None, timesteps, input_params))
y = tf.placeholder(tf.float32, (None, classes))

# LSTM weights
W = {'h_layer': tf.Variable(tf.random_normal([input_params, hidden_units])),'o_layer': tf.Variable(tf.random_normal([hidden_units, classes], mean=1.0))}
b = {'h_layer': tf.Variable(tf.random_normal([hidden_units])),'o_layer': tf.Variable(tf.random_normal([classes], mean = 1.0))}

# Preparing input x for the LSTM network
x_mod = tf.reshape([tf.transpose(x, [1, 0, 2])], [-1, input_params])
pred = LSTM_RNN(x_mod, W, b)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred)
cost = tf.reduce_mean(cross_entropy)

adam = tf.train.AdamOptimizer(learning_rate= 0.0025, beta1 = 0.8, beta2 = 0.8, epsilon = 1e-08)
cost_min = adam.minimize(cost) # Adam Optimizer
argmax_pred = tf.argmax(pred,1)
argmax_y = tf.argmax(y,1)

est_update = tf.cast(tf.equal(argmax_pred, argmax_y), tf.float32)
accuracy = tf.reduce_mean(est_update)

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "n_samples" amount of example data at each loop
step = 1
while step * n_samples <= n_iter:
    batchdatas =         extract_n_samples(X_train, step, n_samples)
    batch_ys = y_wrap_fn(extract_n_samples(y_train, step, n_samples))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [cost_min, cost, accuracy],
        feed_dict={
            x: batchdatas, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*n_samples % d_iter == 0) or (step == 1) or (step * n_samples > n_iter):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training: Iteration #" + str(step*n_samples) + \
              ":   Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: y_wrap_fn(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("Testing: " + \
              "Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

# Accuracy for test data

y_wrap_fn_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: y_wrap_fn(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

    

