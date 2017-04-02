
#%%
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file =   './data/train.p'
validation_file = './data/valid.p'
testing_file =    './data/test.p'

classes_file = 'signnames.csv'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

#%% Normalize the data

#def normalize_image(x):
#    """
#    Normalize a list of sample image data in the range of 0 to 1
#    : x: List of image data.  The image shape is (32, 32, 3)
#    : return: Numpy array of normalize data
#    """
#    x_norm = x.reshape(x.size)
#    
#    x_max = max(x_norm)
#    x_min = min(x_norm)
#    x_range = x_max-x_min
#    x_norm = (x_norm - x_min)/(x_range)    
#    
#    return x_norm.reshape(x.shape)
#
#def normalize_image_list( images ):
#    for image in images:
#        image = normalize_image( image )


#plot_image(1, X_train, y_train)
#normalize_image_list( X_train )
#
#
#plot_image(1, X_train, y_train)


#%%
#
#plot_image(1, X_train, y_train)
#print(X_train[1][1][1:7])
#X_train[1] = normalize_image( X_train[1] )
#print(X_train[1][1][1:7])
#plot_image(1, X_train, y_train)


#%%
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import csv 

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
#label_types = {'num': [], 'meaning': [] }
label_list = []
label_index = []
labels_Names = []

with open(classes_file,'r') as fl:
    reader = csv.reader( fl )
    for row in reader:
        if row[0].isdecimal():
            label_list.append( [row[0],  row[1]]  )
            label_index.append( row[0] )
            labels_Names.append( row[1] )
#            label_types['num'].append( row[0] )
#            label_types['meaning'].append( row[1] )


labels_N = len( label_list )            
print('There are : {} label types.'.format( labels_N ) )
print('Example: ' + str(label_list[0]) )
        
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)


#%% Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random

def plot_image(image_i,  images, label_ids):

    image = images[image_i]
    label_id = label_ids[image_i]
    
    fig, axies = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    fig.suptitle('Image '+str(image_i), fontsize=20, y=1.1)

#    pred_names = [labels_Names[pred_i] for pred_i in pred_indicies]
    correct_name = labels_Names[label_id]

    axies.imshow(image)
    axies.set_title(correct_name)
    axies.set_axis_off()

    axies.barh(0, 0, 0)
#    axies.set_yticks()
#    axies.set_yticklabels(pred_names[::-1])
    axies.set_xticks([0, 0.5, 1.0])


index = random.randint(0, len(X_train))
plot_image(index, X_train, y_train)
index = random.randint(0, len(X_train))
plot_image(index, X_train, y_train)
index = random.randint(0, len(X_train))
plot_image(index, X_train, y_train)

#%% Define the network

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128*2


from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for 
    # the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Normalize input
    x = tf.contrib.layers.batch_norm( x, is_training = True)
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Shape of the filter-weights for the convolution.
    c1_s = [5, 5, 3, 6]

    # Create new weights and bias
    c1_w = tf.Variable( tf.truncated_normal(c1_s, mean = mu, stddev = sigma), name = 'c1_w' )
    c1_b = tf.Variable( tf.zeros(6), name = 'c1_b' )

    # Create the TensorFlow operation for convolution.
    c1 = tf.nn.conv2d(  input = x,
                        filter = c1_w,
                        strides = [1, 1, 1, 1],
                        padding = 'VALID')
    c1 = tf.nn.bias_add( c1, c1_b)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    c1 = tf.nn.max_pool(    value =   c1,
                            ksize =   [1, 2, 2, 1],
                            strides = [1, 2, 2, 1],
                            padding = 'SAME')
    # Activation.
    c1 = tf.nn.relu( c1 )
    c1 = tf.contrib.layers.batch_norm( c1, is_training = True)
    

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # Shape of the filter-weights for the convolution.
    c2_s = [5, 5, 6, 16]

    # Create new weights and bias
    c2_w = tf.Variable( tf.truncated_normal(c2_s, mean = mu, stddev = sigma), name = 'c2_w' )
    c2_b = tf.Variable( tf.zeros( c2_s[3] ), name = 'c2_b' )

    # Create the TensorFlow operation for convolution.
    c2 = tf.nn.conv2d(  input = c1,
                        filter = c2_w,
                        strides = [1, 1, 1, 1],
                        padding = 'VALID')
    c2 = tf.nn.bias_add( c2, c2_b)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    c2 = tf.nn.max_pool(    value =   c2,
                            ksize =   [1, 2, 2, 1],
                            strides = [1, 2, 2, 1],
                            padding = 'SAME')
    # Activation.
    c2 = tf.nn.relu( c2 )
    c2 = tf.contrib.layers.batch_norm( c2, is_training = True)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat = flatten( c2 )
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_s = (400, 120)
    fc1_w = tf.Variable( tf.truncated_normal(fc1_s, mean = mu, stddev = sigma), name = 'fc1_w' )
    fc1_b = tf.Variable( tf.zeros(120), name = 'fc1_b' )
    fc1 = tf.add( tf.matmul( flat, fc1_w ), fc1_b)
    
    fc1 = tf.nn.relu( fc1 )
    fc1 = tf.nn.dropout( fc1, keep_prob )
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_s = (120, 84)
    fc2_w = tf.Variable( tf.truncated_normal(fc2_s, mean = mu, stddev = sigma), name = 'fc2_w' )
    fc2_b = tf.Variable( tf.zeros(84), name = 'fc2_b' )
    fc2 = tf.add( tf.matmul( fc1, fc2_w ), fc2_b)
    
    fc2 = tf.nn.relu( fc2 )
    fc2 = tf.nn.dropout( fc2, keep_prob )
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = labels_N.
    fc3_s = (84, labels_N)
    fc3_w = tf.Variable( tf.truncated_normal(fc3_s, mean = mu, stddev = sigma), name = 'fc3_w' )
    fc3_b = tf.Variable( tf.zeros(labels_N), name = 'fc3_b' )
    logits = tf.add( tf.matmul( fc2, fc3_w ), fc3_b)
    
    return logits



x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'x')
y = tf.placeholder(tf.int32, (None), name = 'y')
one_hot_y = tf.one_hot(y, labels_N, name = 'one_hot_y')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')


rate = 0.001
logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer( learning_rate = rate, name = 'optimizer' )
training_operation = optimizer.minimize( loss_operation, name = 'training_operation' )


#%% Evaluate accuracy funtion

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples

#%% Training

# Dropout keep probability 
keep_prob_input = 0.5
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_input})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} - Validation Accuracy = {:.1%}".format(i+1, validation_accuracy))
        
    saver.save(sess, './lenet')
    print("Model saved")
    print()

#%% Re-training
if 0:
    #%%
    # Dropout keep probability 
    EPOCHS = 10
    keep_prob_input = 0.5
        
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        print("Model loaded")
        num_examples = len(X_train)
        
        print("Re-Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_input})
                
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} - Validation Accuracy = {:.1%}".format(i+1, validation_accuracy))
            
        saver.save(sess, './lenet')
        print("Model saved")
        print()

#%% Test accuracy

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.1%}".format(test_accuracy))

