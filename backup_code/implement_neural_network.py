"""
@uthor: Himaghna, 27th September, 2019
Description: Implement neural network to predict E/Z
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import data_processing
from data_post_process import bin_to_families, get_family_membership_idx, \
    get_train_test_idx
from helper_functions import plot_parity



processed_data = data_processing.main()
X, y = processed_data['X'], processed_data['y']

bin_bounds = [2, 5]
n_bins = len(bin_bounds) + 1
descriptor_names = processed_data['descriptor_names']
family_one_hotx = processed_data['family_one_hotx']
y, y_ordinal = bin_to_families(y, bin_upper_bounds=bin_bounds)
# get class membership ids to split to train/ test independently in each set
class_membership_idx = get_family_membership_idx(y_ordinal)
# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


def do_nn(**kwargs):
    """
    Fit a neural network to the data-set
    Params ::
    **kwargs: optional keyworks
    Returns ::
    None
    """
    args = dict()
    args['train_size'] = 0.9
    args['random_state'] = 23
    args.update(**kwargs)

    x_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y_ordinal, train_size=args['train_size'])
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int64)
        return x, y


    def create_dataset(xs, ys, n_classes=n_bins):
        ys = tf.one_hot(ys, depth=n_classes)
        print(xs)
        return tf.data.Dataset.from_tensor_slices((xs, ys)) \
            .map(preprocess) \
                .shuffle(len(ys)) \
                    .batch(5)

    train_dataset = create_dataset(X_std_train, y_train)
    val_dataset = create_dataset(X_std_test, y_test)
    # create model
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(15, ), input_shape=(15,)),
        keras.layers.Dense(units=20, activation='relu'),
       # keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=n_bins, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
          train_dataset.repeat(),
          epochs=100,
          steps_per_epoch=6,
          validation_data=val_dataset.repeat(),
          validation_steps=1
    )


do_nn()



