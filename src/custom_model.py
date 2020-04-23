from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
 
def rgb_to_gray(image_array,bgr=False):
    if bgr: return np.array([[0.21*image_array[i][j][2] + 0.72*image_array[i][j][1] 
        + 0.07*image_array[i][j][0] for j in range(image_array.shape[1])] for i in range(image_array.shape[0])])
    return np.array([[0.21*image_array[i][j][0] + 0.72*image_array[i][j][1] 
        + 0.07*image_array[i][j][2] for j in range(image_array.shape[1])] for i in range(image_array.shape[0])])

class custom_model():

    save_path="../data/models/"
    model_name="binary_class_chess_square.h5"

    # dimensions of our images.
    img_width, img_height = 32, 32

    def __init__(self):
        self.model=Sequential()
        
    def compile(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 1)

        self.model.add(Conv2D(16, (3, 3), input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def load_model(self,model_location=save_path+model_name):
        self.compile()
        self.model.load_weights(model_location)

    # will return classification value 
    # 1 : unpopulated
    # 0 : populated
    def run_inference(self,img):
        img=rgb_to_gray(img,bgr=True)
        img=np.expand_dims(img, axis=-1)
        img=np.expand_dims(img, axis=0)
        return self.model.predict_proba(img)[0][0]

