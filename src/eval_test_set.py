import argparse
import numpy as np
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from model import get_model
from dataset_utils import get_dataset, train_val_split

# Seaborn settings
sn.set(style="whitegrid", font_scale=1.5)
sn.set_palette("bright", color_codes=True)

# CLI
PARSER = argparse.ArgumentParser(description='CLI for training pipeline')
PARSER.add_argument('--batch_size', type=int, default=160, help='Batch size per step')
PARSER.add_argument('--init_weight', type=str, default=None, help='Path to trained weights')
ARGS = PARSER.parse_args()

# Create test set
BATCH_SIZE = ARGS.batch_size
print('Creating test dataset...')
_, TEST_FILES = train_val_split()
#TEST_FILES = glob('ModelNet40/*/test/*.npy')   # only used to get length for comparison
print('Number of test samples:', len(TEST_FILES))
#print('Number of testing samples:', len(TEST_FILES))

print(TEST_FILES.iloc[10])
test_ds = get_dataset(TEST_FILES, BATCH_SIZE)
print('Dataset ready!')

#Class look-ups
idx_lookup = {
    'proton':0
    ,'pi-':1 
    ,'e-':2
    #,'k+'
    ,'mu-':3
    ,'pion0':4
    }


# Instantiate metrics
accuracy = tf.keras.metrics.CategoricalAccuracy()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
false_neg = tf.keras.metrics.FalseNegatives()
false_pos = tf.keras.metrics.FalsePositives()
true_neg = tf.keras.metrics.TrueNegatives()
true_pos = tf.keras.metrics.TruePositives()


# Load model
model = get_model(bn_momentum=None)
latest_ckpt = tf.train.latest_checkpoint(ARGS.init_weight)
print('Loading weights from: ',latest_ckpt)
model.load_weights(latest_ckpt)


# Evaluate
size = 0
confusion_mat = np.zeros((5, 5), dtype=np.float32)
for x_test, y_test in test_ds:
    size += x_test.shape[0]

    logits = model(x_test, training=False)
    probs = tf.math.sigmoid(logits)

    accuracy.update_state(y_test, probs)

    max_idxs = tf.math.argmax(probs, axis=1)
    one_hot = tf.one_hot(max_idxs, depth=5, dtype=tf.float32)
    precision.update_state(y_test, one_hot)
    recall.update_state(y_test, one_hot)
    false_neg.update_state(y_test, one_hot)
    false_pos.update_state(y_test, one_hot)
    true_neg.update_state(y_test, one_hot)
    true_pos.update_state(y_test, one_hot)

    for true_class, pred_class in zip(np.argwhere(y_test)[:, 1], max_idxs.numpy()):
        confusion_mat[true_class, pred_class] += 1


print('Test set size =', size)
print('Accuracy\t', accuracy.result().numpy())
print('Precision\t', precision.result().numpy())
print('Recall\t\t', recall.result().numpy())
print('False negatives\t', false_neg.result().numpy())
print('False positives\t', false_pos.result().numpy())
print('True negatives\t', true_neg.result().numpy())
print('True positives\t', true_pos.result().numpy())


# Save confusion matrix
row_norm = np.sum(confusion_mat, axis=1)
row_norm = np.expand_dims(row_norm, axis=1)
row_norm = np.repeat(row_norm, 5, axis=1)
confusion_mat /= row_norm

mask = confusion_mat < 0.05
plt.figure(figsize=(11, 10.5))  # width by height
ax = sn.heatmap(confusion_mat, annot=True, annot_kws={'size': 9},
                fmt='.1f', cbar=False, cmap='binary', mask=mask, linecolor='black', linewidths=0.5)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_ylabel('True Class')
ax.set_xlabel('Predicted Class')
ax.set_xticklabels(list(idx_lookup.keys()))
ax.set_yticklabels(list(idx_lookup.keys()))
ax.spines['top'].set_visible(True)
plt.yticks(rotation=0)
plt.savefig('figs/confusion_matrix.png')
