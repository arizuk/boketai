import tensorflow as tf
import os
import math

import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Activation, Flatten
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras._impl.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras._impl.keras.utils import np_utils
from IPython.display import Image, display, SVG
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda
from PIL import Image

def sort_data_by_length(data_x, data_y):
    data_x_lens = [len(datum) for datum in data_x]
    sorted_data_indexes = sorted(range(len(data_x_lens)), key=lambda x: -data_x_lens[x])

    data_x = [data_x[i] for i in sorted_data_indexes]
    data_y = [data_y[i] for i in sorted_data_indexes]

    return data_x, data_y

def encode(sentence, w2i, unk='<unk>'):
    return [w2i[w] if w in w2i else w2i[unk] for w in sentence]

def build_w2i(path):
    # Build w2i & i2w
    vocab = set()
    for line in open(path):
        sentence = line.strip().split()
        vocab.update(sentence)
    w2i = {w: np.int32(i+4) for i, w in enumerate(vocab)}
    w2i['<s>'], w2i['</s>'], w2i['<unk>'], w2i['<pad>'] = \
        np.int32(0), np.int32(1), np.int32(2), np.int32(3)
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

def build_dataset(path, w2i):
    # Encode data with w2i
    data = []
    for line in open(path):
        sentence = line.strip().split()
        sentence = ['<s>'] + sentence + ['</s>']
        encoded_sentence = encode(sentence, w2i)
        data.append(encoded_sentence)

    return data

def save_model(model, name):
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    result_dir = os.path.normpath(data_dir)
    model.save_weights(os.path.join(result_dir, name + '_model.h5'))

def load_weight(model, name):
    data_dir = 'data'
    result_dir = os.path.normpath(data_dir)
    weight_file = os.path.join(result_dir, name + '_model.h5')
    model.load_weights(weight_file)
    return model

    return model

def generator(data_x, data_y, batch_size=32):

    n_batches = math.ceil(len(data_x) / batch_size)

    while True:
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_x_mb = data_x[start:end]
            data_y_mb = data_y[start:end]

            data_x_mb = np.array(data_x_mb).astype('float32') / 255.
            data_y_mb = pad_sequences(data_y_mb, dtype='int32', padding='post', value=w2i['<pad>'])
            data_y_mb_oh = np.array([np_utils.to_categorical(datum_y, vocab_size) for datum_y in data_y_mb[:, 1:]])

            yield [data_x_mb, data_y_mb], data_y_mb_oh

def load_images(file):
    f = open(file)
    images = []
    for line in f:
        path = "data/images/"  + line.rstrip()
        img = Image.open(path)
        img = img.resize((224, 224))
        images.append(np.array(img))
    f.close()
    return np.array(images)

if __name__ == '__main__':
    np.random.seed(34)

    train_x = load_images('data/images.v1')
    valid_x = train_x[:100]
    train_x = train_x[100:]

    w2i, i2w = build_w2i('data/texts.v1')
    train_y = build_dataset('data/texts.v1', w2i)
    # valid_y = build_dataset('data/texts.v1', w2i)
    valid_y = train_y[:100]
    train_y = train_y[100:]

    vocab_size = len(w2i)

    # キャプションの長さでソート (理由は後述)
    train_y, train_x = sort_data_by_length(train_y, train_x)
    valid_y, valid_x = sort_data_by_length(valid_y, valid_x)

    K.clear_session()

    emb_dim = 128
    hid_dim = 128
    batch_size = 32

    x = Input(shape=(224, 224, 3))
    encoder = VGG16(weights='imagenet', include_top=False, input_tensor=x)

    # パラメータを固定
    for layer in encoder.layers:
        layer.trainable = False

    # CNNの出力
    u = Flatten()(encoder.output)

    # LSTMの初期状態
    h_0 = Dense(hid_dim)(u)
    c_0 = Dense(hid_dim)(u)

    # LSTMの入力
    y = Input(shape=(None,), dtype='int32')
    y_in = Lambda(lambda x: x[:, :-1])(y)
    y_out = Lambda(lambda x: x[:, 1:])(y)

    # 誤差関数のマスク
    mask = Lambda(lambda x: K.cast(K.not_equal(x, w2i['<pad>']), 'float32'))(y_out)

    # 層の定義
    embedding = Embedding(vocab_size, emb_dim)
    lstm = LSTM(hid_dim, activation='tanh', return_sequences=True, return_state=True)
    dense = Dense(vocab_size)
    softmax = Activation('softmax')

    # 順伝播
    y_emb = embedding(y_in)
    h, _, _ = lstm(y_emb, initial_state=[h_0, c_0]) # 第2,3戻り値(最終ステップのh, c)は無視
    h = dense(h)
    y_pred = softmax(h)

    def masked_cross_entropy(y_true, y_pred):
        return -K.mean(K.sum(K.sum(y_true * K.log(K.clip(y_pred, 1e-10, 1)), axis=-1) * mask, axis=1))

    model = Model([x, y], y_pred)
    model.compile(loss=masked_cross_entropy, optimizer='rmsprop')

    # # load weight
    # if os.path.exists('data/image2text_model.h5'):
    #     model = load_weight(model, 'image2text')

    n_batches_train = math.ceil(len(train_x) / batch_size)
    n_batches_valid = math.ceil(len(valid_x) / batch_size)

    model.fit_generator(
        generator(train_x, train_y, batch_size),
        epochs=1000,
        steps_per_epoch=n_batches_train,
        validation_data=generator(valid_x, valid_y),
        validation_steps=n_batches_valid
    )

    encoder_model = Model([x], [h_0, c_0])

    # 入力
    h_tm1 = Input(shape=(hid_dim,))
    c_tm1 = Input(shape=(hid_dim,))
    y_t = Input(shape=(1,))

    # 順伝播
    y_emb_t = embedding(y_t)
    _, h_t, c_t = lstm(y_emb_t, initial_state=[h_tm1, c_tm1])
    pred_t = dense(h_t)
    pred_t = softmax(pred_t)

    decoder_model = Model([y_t, h_tm1, c_tm1], [pred_t, h_t, c_t])

    def decode_sequence(x, max_len=100):
        h_tm1, c_tm1 = encoder_model.predict(x)
        y_tm1 = np.array([w2i['<s>']])
        y_pred = np.array([w2i['<s>']])

        while True:
            y_t, h_t, c_t = decoder_model.predict([y_tm1, h_tm1, c_tm1])
            y_t = np.argmax(y_t.flatten())
            y_pred = np.append(y_pred, [y_t])

            h_tm1, c_tm1 = h_t, c_t
            y_tm1 = y_pred[-1:]

            if y_pred[-1] == w2i['</s>'] or len(y_pred) > max_len:
                break
        return y_pred

    test_x = np.array(valid_x[:1])
    pred_y = decode_sequence(test_x)
    print(' '.join([i2w[i] for i in pred_y]))

    # plt.imshow(test_x[0])

    # save model
    save_model(model, 'image2text')