import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
IMAGE_SIZE = 96
NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
NUM_LABELS = 30
NUM_EPOCHS = 200
VALIDATION_SIZE = 100  # Size of the validation set.
EARLY_STOP_PATIENCE = 100


def plot_sample(x, y, truth=None):
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.scatter(y[0::2] * 96, y[1::2] * 96)
    if truth is not None:
        plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
    plt.savefig("data/img.png")


def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def plot_learning_curve(loss_train_record, loss_valid_record):
    plt.figure()
    plt.plot(loss_train_record, label='train')
    plt.plot(loss_valid_record, c='r', label='validation')
    plt.ylabel("RMSE")
    plt.legend(loc='upper left', frameon=False)
    plt.savefig("data/learning_curve.png")


def generate_submission(test_dataset, sess, eval_prediction, eval_data_node):
    test_labels = eval_in_batches(test_dataset, sess, eval_prediction, eval_data_node)
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    cols = joblib.load('data/cols.pkl')

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('data/submission.csv', index=False)

def error_measure(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / (2 * predictions.shape[0])


#讀檔
train_data = pd.read_csv(FTRAIN)
cols = train_data.columns[:-1]
train_data['Image'] = train_data['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
train_data = train_data.dropna()
dataset = np.vstack(train_data['Image'])
dataset = dataset.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
labels = train_data[cols].values / 96.0
dataset, labels = shuffle(dataset, labels)
joblib.dump(cols, 'data/cols.pkl', compress=3)

test_data = pd.read_csv(FTEST)
test_data['Image'] = test_data['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
test_data = test_data.dropna()
test_dataset = np.vstack(test_data['Image'])
test_dataset = test_dataset.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

#數據切分
validation_dataset, train_dataset = dataset[:VALIDATION_SIZE, ...], dataset[VALIDATION_SIZE:, ...]
validation_labels, train_labels = labels[:VALIDATION_SIZE], labels[VALIDATION_SIZE:]
train_size = train_labels.shape[0]
print("training size is %d" % train_size)

if __name__ == '__main__':
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    eval_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1, seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc3_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    def model(data, train=False):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc3_weights) + fc3_biases

    train_prediction = model(train_data_node, True)

    # Minimize the squared errors
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction - train_labels_node), 1))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
    # Add the regularization term to the loss.
    loss += 1e-7 * regularizers
    eval_prediction = model(eval_data_node)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1e-3, global_step * BATCH_SIZE, train_size, 0.95, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

    # tensorboard
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())


    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    loss_train_record = list() # np.zeros(n_epoch)
    loss_valid_record = list() # np.zeros(n_epoch)

    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0
    current_epoch = 0

    while current_epoch < NUM_EPOCHS:
        # Shuffle data
        shuffled_index = np.arange(train_size)
        np.random.shuffle(shuffled_index)
        train_dataset = train_dataset[shuffled_index]
        train_labels = train_labels[shuffled_index]

        for step in range(int(train_size / BATCH_SIZE)):
            offset = step * BATCH_SIZE
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, loss_train, current_learning_rate = sess.run([train_step, loss, learning_rate], feed_dict=feed_dict)

        # After one epoch, make validation
        eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)
        loss_valid = error_measure(eval_result, validation_labels)

        print ('Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f' % (
            current_epoch,
            loss_train, loss_valid,
            loss_train / loss_valid,
            current_learning_rate
        ))
        loss_train_record.append(np.log10(loss_train))
        loss_valid_record.append(np.log10(loss_valid))

        if loss_valid < best_valid:
            best_valid = loss_valid
            best_valid_epoch = current_epoch
        elif best_valid_epoch + EARLY_STOP_PATIENCE < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
            break

        current_epoch += 1

    print('train finish')

    generate_submission(test_dataset, sess, eval_prediction, eval_data_node)

    # Show an example of comparison
    i = 0
    img = validation_dataset[i]
    lab_y = validation_labels[i]
    lab_p = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)[0]
    plot_sample(img, lab_p, lab_y)
    plot_learning_curve(loss_train_record, loss_valid_record)
