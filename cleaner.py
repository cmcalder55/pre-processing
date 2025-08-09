from tensorflow.keras.utils import to_categorical

def normalize_img_data(X_train, y_train, X_test, y_test):
    # flatten 28*28 images to a 784 vector for each image
    n_pix = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], n_pix).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], n_pix).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train /= 255
    X_test /= 255

    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)
