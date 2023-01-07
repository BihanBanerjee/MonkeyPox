import tensorflow as tf
from tensorflow import keras
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, Flatten
from keras import layers
from keras.models import Model

def create_model(fold_no, model_name,IMG_SIZE = 224, output = 2):

    print('------------------------------------------------------------------------')
    print()
    print("fold no --- " , fold_no)
    print()
    print('------------------------------------------------------------------------')

    print()
    print(f"Model ------- {model_name}")
    print()
    
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  
    if(model_name == "Xception" ):

        model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
    elif(model_name == "InceptionV3"):
        model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                                                        include_top=False,
                                                                        weights='imagenet')
        
    elif(model_name == "DenseNet169"):
        model = tf.keras.applications.DenseNet169(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet')
    else:
        return        


    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
    gaussian_input = layers.GaussianNoise(0.01)(inputs)
    base_layer = model(gaussian_input)
    dropout_layer_1 = layers.Dropout(0.0)(base_layer)
    flat_layer = layers.Flatten()(dropout_layer_1)
    dense_1 = layers.Dense(512, activation="relu")(flat_layer)
    dropout_layer_2 = layers.Dropout(0.4)(dense_1)
    dense_2 = layers.Dense(256, activation="relu")(dropout_layer_2)
    dropout_layer_3 = layers.Dropout(0.3)(dense_2)
    dense_3 = layers.Dense(128, activation="relu", name="extraction")(dropout_layer_3)
    dropout_layer_4 = layers.Dropout(0.2)(dense_3)
    outputs = layers.Dense(output, activation="softmax")(dropout_layer_4)



    model = Model(inputs=model.input, outputs=outputs)

    my_model = tf.keras.models.clone_model(model)
    return my_model