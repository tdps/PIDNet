import argparse
from glob import glob
from time import time
from datetime import timezone, datetime, timedelta
import os

import tensorflow as tf

from model import get_model
from losses import anchor_loss
from dataset_utils import get_dataset, train_val_split

def get_timestamp():
    timestamp = str(datetime.now(timezone.utc))[:16]
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(' ', '_')
    timestamp = timestamp.replace(':', '')
    return timestamp

tf.random.set_seed(0)


# CLI
PARSER = argparse.ArgumentParser(description='CLI for training pipeline')
PARSER.add_argument('--batch_size', type=int, default=64, help='Batch size per step')
PARSER.add_argument('--epochs', type=int, default=50, help='Number of epochs')
PARSER.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
PARSER.add_argument('--init_weight', type=str, default=None, help='Path to initial weights')
ARGS = PARSER.parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.learning_rate
#LR_DECAY_STEPS = 10000
#LR_DECAY_RATE = 0.7
INIT_TIMESTAMP = get_timestamp()
LOG_DIR = 'logs'

# Create datasets (.map() after .batch() due to lightweight mapping fxn)
print('Creating train and val datasets...')
f_train = '/home/yalmalioglu/dataset5d/1500sp_padding_evts/train_files35k_5p.csv'
f_test = '/home/yalmalioglu/dataset5d/1500sp_padding_evts/test_files35k_5p.csv'

TRAIN_FILES, VAL_FILES = train_val_split(f_train,f_test)
#TEST_FILES = glob('ModelNet40/*/test/*.npy')   # only used to get length for comparison
print('Number of training samples:', len(TRAIN_FILES))
print('Number of validation samples:', len(VAL_FILES))
#print('Number of testing samples:', len(TEST_FILES))

print(TRAIN_FILES.iloc[5])
train_ds = get_dataset(TRAIN_FILES, BATCH_SIZE)
val_ds = get_dataset(VAL_FILES, BATCH_SIZE)
print('Datasets ready!')


# Create model
def get_bn_momentum(step):
    #return min(0.99, 0.5 + 0.0002*step)
    return 0.99
print('Creating model...')
bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)
model = get_model(bn_momentum=bn_momentum)
print('Model ready!')
model.summary()

if ARGS.init_weight:
    latest_ckpt = tf.train.latest_checkpoint(ARGS.init_weight)
    print('Resuming training from: ',latest_ckpt)
    model.load_weights(latest_ckpt)


# Instantiate optimizer and loss function
def get_lr(initial_learning_rate, decay_steps, decay_rate, step, staircase=False, warm_up=True):
    if warm_up:
        coeff1 = min(1.0, step/2000)
    else:
        coeff1 = 1.0

    if staircase:
        coeff2 = decay_rate ** (step // decay_steps)
    else:
        coeff2 = decay_rate ** (step / decay_steps)

    current = initial_learning_rate * coeff1 * coeff2
    return current
#LR_ARGS = {'initial_learning_rate': LEARNING_RATE, 'decay_steps': LR_DECAY_STEPS,
#           'decay_rate': LR_DECAY_RATE, 'staircase': False, 'warm_up': True}
#lr = tf.Variable(get_lr(**LR_ARGS, step=0), trainable=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fxn = anchor_loss


# Instantiate metric objects
train_acc = tf.keras.metrics.CategoricalAccuracy()
train_prec = tf.keras.metrics.Precision()
train_recall = tf.keras.metrics.Recall()
val_acc = tf.keras.metrics.CategoricalAccuracy()
val_prec = tf.keras.metrics.Precision()
val_recall = tf.keras.metrics.Recall()


# Training
print('Training...')
print('Steps per epoch =', len(TRAIN_FILES) // BATCH_SIZE)
print('Total steps =', (len(TRAIN_FILES) // BATCH_SIZE) * EPOCHS)

@tf.function
def train_step(inputs, labels):
    # Forward pass with gradient tape and loss calc
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_fxn(labels, logits) + sum(model.losses)

    # Obtain gradients of trainable vars w.r.t. loss and perform update
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss #, model.losses[0]

@tf.function
def val_step(inputs):
    logits = model(inputs, training=False)
    return logits


# Instantiate log writers for tensorboard    
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(LOG_DIR, 'gradient_tape', current_time, 'train')
test_log_dir = os.path.join(LOG_DIR, 'gradient_tape', current_time, 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

training_tic = time()
step = 0
for epoch in range(EPOCHS):
    print('\nEpoch', epoch)

    epoch_tic = time()

    # Reset metrics
    train_acc.reset_states()
    train_prec.reset_states()
    train_recall.reset_states()
    val_acc.reset_states()
    val_prec.reset_states()
    val_recall.reset_states()
    
    #print('\n Reset Metrics')

    # Train on batches
    for x_train, y_train in train_ds:
        step_tic = time()

        #print("x_train shape: ",x_train.shape)
        #print("y_train shape: ",y_train.shape)
        
        #train_logits, train_loss, mat_reg_loss = train_step(x_train, y_train)
        train_logits, train_loss = train_step(x_train, y_train)

        train_probs = tf.math.sigmoid(train_logits)
        train_acc.update_state(y_train, train_probs)

        max_idxs = tf.math.argmax(train_probs, axis=1)
        train_one_hot = tf.one_hot(max_idxs, depth=5, dtype=tf.float32)
        train_prec.update_state(y_train, train_one_hot)
        train_recall.update_state(y_train, train_one_hot)


        # Collect statistics for tensorboard  
        with train_summary_writer.as_default():
            tf.summary.scalar('train_accuracy', train_acc.result(), step=step)
            tf.summary.scalar('train_precision', train_prec.result(), step=step)
            tf.summary.scalar('train_recall', train_recall.result(), step=step)

        step += 1
        bn_momentum.assign(get_bn_momentum(step))
        #lr.assign(get_lr(**LR_ARGS, step=step))

    # Run validation on batches at the end of epoch
    for x_val, y_val in val_ds:

        # print("x_val shape: ",x_val.shape)
        # print("y_val shape: ",y_val.shape)

        val_logits = val_step(x_val)

        val_probs = tf.math.sigmoid(val_logits)
        val_acc.update_state(y_val, val_probs)

        max_idxs = tf.math.argmax(val_probs, axis=1)
        val_one_hot = tf.one_hot(max_idxs, depth=5, dtype=tf.float32)
        val_prec.update_state(y_val, val_one_hot)
        val_recall.update_state(y_val, val_one_hot)

        # Collect statistics for tensorboard
        with test_summary_writer.as_default():
            tf.summary.scalar('val_accuracy', val_acc.result(), step=step)
            tf.summary.scalar('val_precision', val_prec.result(), step=step)
            tf.summary.scalar('val_recall', val_recall.result(), step=step)

    # Save every 10 epoch
    if epoch>0 and epoch % 25 == 0:
        print('Saving weights at step :{}, epoch: {}'.format(step, epoch))
        model.save_weights(os.path.join(LOG_DIR, 'checkpoints', current_time, 'iter-' + str(step)), save_format='tf')
    
    #Print statistics
    template = 'Epoch {}, Time per epoch: {}, Accuracy: {}, Precision: {}, Test Accuracy: {}, Test Precision: {}'
    print (template.format(epoch,
                        str(timedelta(seconds=time()-epoch_tic)),
                         train_acc.result()*100, 
                         train_prec.result(),
                         val_acc.result()*100, 
                         val_prec.result()))

print('Saving final weights :{}, epoch: {}'.format(step, epoch))
model.save_weights(os.path.join(LOG_DIR, 'checkpoints', current_time, 'iter-' + str(step)), save_format='tf')

print('Done training!')
print('Total time for training: ', str(timedelta(seconds=time()-training_tic)))
