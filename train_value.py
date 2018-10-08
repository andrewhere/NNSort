from __future__ import print_function

import generate_data as gd
import tensorflow as tf
import sortedness as st
import time

def train(array_start, array_end, num_input, num_array, num_test):
    # Parameters
    learning_rate = 0.1
    num_steps = 6000
    batch_size = 200
    display_step = 100

    # Network Parameters
    n_hidden_1 = 40 # 1st layer number of neurons
    n_hidden_2 = 40 # 2nd layer number of neurons
    #num_input = 5 # MNIST data input (img shape: 28*28)
    num_classes = num_input # MNIST total classes (0-9 digits)
    #array_start = 0
    #array_end = 10
    #num_array = 2000

    # tf Graph input
    X = tf.placeholder("float32", [None, num_input])
    Y = tf.placeholder("float32", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal(shape=[num_input, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2],stddev=0.1)),
        'out': tf.Variable(tf.random_normal(shape=[n_hidden_2, num_classes], stddev=0.1))
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        'out': tf.Variable(tf.constant(0.1, shape=[num_classes]))
    }


    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = (tf.matmul(x, weights['h1']) + biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = (tf.matmul(layer_1, weights['h2']) + biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = (logits)

    # Define loss and optimizer
    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #    logits=logits, labels=Y))
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=Y, predictions=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.contrib.framework.argsort(prediction), tf.contrib.framework.argsort(Y))
    #correct_pred = tf.equal(tf.convert_to_tensor(prediction), tf.convert_to_tensor(Y))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    #generate data and get labels
    array = gd.gen_array(num_input, array_start, array_end, num_array)
    normalized_array = gd.normalize(array_start, array_end, array)
    labels = gd.sorted_array(normalized_array)
    arg_labels = gd.argsort_list(array)
    test_x = gd.gen_array(num_input, array_start, array_end, num_test)
    norm_test_x = gd.normalize(array_start, array_end, test_x)
    test_y = gd.sorted_array(norm_test_x)
    test_y_arg = gd.argsort_list(test_x)
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        tot_acc = 0
        loop_count = 0
        #set timer
        start = time.time()

        for step in range(1, num_steps+1):
            #print("Weight: ", sess.run(weights['h1']))
            #print("Bias: ", sess.run(biases['b1']))
            batch_x, batch_y = gd.next_batch(batch_size, normalized_array, labels)
            # Run optimization op (backprop)
            #print("Array: ", batch_x)
            #print("L1: ", sess.run(logits, feed_dict={X: batch_x, Y: batch_y}))
            #print("X = ", sess.run(X, feed_dict= {X:batch_x}))
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                #print("Pred: ", sess.run(l2,feed_dict={X: batch_x, Y: batch_y}))
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                tot_acc += acc
                loop_count += 1
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        end = time.time()
        print("Training Complete, Time Used: ", (end-start))
        print("Averaged Training Accuray: ", tot_acc/loop_count)

        print("Optimization Finished!")
        #test_x = [[0.3, 0.9, 0.7, 0.4, 0.1],[0.6, 0.3, 0.1, 0.9, 0.5], [0.2, 0.4, 0.6, 0.9, 0.7]]

        pred, pred_acc = sess.run([prediction, accuracy], feed_dict = {X: norm_test_x, Y:test_y})
        print("Expected Output: ", test_y)

    return pred, pred_acc


prediction, test_acc = train(0, 10, 5, num_array=100000, num_test=3)
print("Test Pred: ", prediction)
#get score of sortedness
bad_score = 0
bad_count = 0
for i in range(len(prediction)):
    score = st.get_score(list(prediction[i]))
    if (score < 1.0):
        print(score)
        bad_score += score
        bad_count += 1

print("Test Accuracy: ", test_acc)
print("Average Score of Sortedness for Bad Outputs: ", bad_score/bad_count)
