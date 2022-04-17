import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import shrunk_covariance


def load_mnist(path, kind='train'):
    # MNISTデータをpathからロード
    # 引数に指定したパスを結合
    label_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    # ファイルを読み込む
    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # 画像ピクセル情報の配列のサイズを変更
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images/255.)-.5)*2
    return images, labels


X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# subplotsで描画を設定：引数で描画領域の行数/列数,x/y軸の統一を指定
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
# 配列を1次元に変形
ax = ax.flatten()
# 配列を28×28に変形
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

np.savez_compressed('mnist_scaled.npz',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)

mnist = np.load('mnist_scaled.npz')

mnist.files
X_train = mnist['X_train']
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
