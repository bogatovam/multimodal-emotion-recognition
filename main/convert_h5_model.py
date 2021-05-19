import logging
from numbers import Real

import tensorflow as tf
from kapre.time_frequency import Melspectrogram
import numpy as np
import soundfile as sf
import resampy
from keras import Input

TARGET_SR = 48000

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, Conv3D
from keras.optimizers import SGD


def create_model_sequential():
    """ Creates model object with the sequential API:
    https://keras.io/models/sequential/
    """

    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model


def create_model_functional():
    """ Creates model object with the functional API:
     https://keras.io/models/model/
     """
    inputs = Input(shape=(16, 112, 112, 3,))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(conv2)

    conv3a = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3a')(pool2)
    conv3b = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3b')(conv3a)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(conv3b)

    conv4a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4a')(pool3)
    conv4b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4b')(conv4a)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(conv4b)

    conv5a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5a')(pool4)
    conv5b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5b')(conv5a)
    zeropad5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                             name='zeropad5')(conv5b)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5')(zeropad5)

    flattened = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flattened)
    dropout1 = Dropout(rate=0.5)(fc6)

    fc7 = Dense(4096, activation='relu', name='fc7')(dropout1)
    dropout2 = Dropout(rate=0.5)(fc7)

    predictions = Dense(487, activation='softmax', name='fc8')(dropout2)

    return Model(inputs=inputs, outputs=predictions)


def create_features_exctractor(C3D_model, layer_name='fc6'):
    extractor = Model(inputs=C3D_model.input,
                      outputs=C3D_model.get_layer(layer_name).output)
    return extractor


if __name__ == "__main__":
    # model = create_model_functional()
    # try:
    #     model.load_weights('../models/pretrained/sports1M_weights_tf.h5')
    # except OSError as err:
    #     print('Check path to the model weights\' file!\n\n', err)
    #
    # import cv2
    # import numpy as np
    #
    # cap = cv2.VideoCapture('videoplayback.mp4')
    #
    # vid = []
    # while True:
    #     ret, img = cap.read()
    #     if not ret:
    #         break
    #     vid.append(cv2.resize(img, (171, 128)))
    # vid = np.array(vid, dtype=np.float32)
    #
    # X = vid[2000:2016, 8:120, 30:142, :]
    # output = model.predict_on_batch(np.array([X]))
    #
    # labels = ["abseiling","air drumming","answering questions","applauding","applying cream","archery","arm wrestling","arranging flowers","assembling computer","auctioning","baby waking up","baking cookies","balloon blowing","bandaging","barbequing","bartending","beatboxing","bee keeping","belly dancing","bench pressing","bending back","bending metal","biking through snow","blasting sand","blowing glass","blowing leaves","blowing nose","blowing out candles","bobsledding","bookbinding","bouncing on trampoline","bowling","braiding hair","breading or breadcrumbing","breakdancing","brush painting","brushing hair","brushing teeth","building cabinet","building shed","bungee jumping","busking","canoeing or kayaking","capoeira","carrying baby","cartwheeling","carving pumpkin","catching fish","catching or throwing baseball","catching or throwing frisbee","catching or throwing softball","celebrating","changing oil","changing wheel","checking tires","cheerleading","chopping wood","clapping","clay pottery making","clean and jerk","cleaning floor","cleaning gutters","cleaning pool","cleaning shoes","cleaning toilet","cleaning windows","climbing a rope","climbing ladder","climbing tree","contact juggling","cooking chicken","cooking egg","cooking on campfire","cooking sausages","counting money","country line dancing","cracking neck","crawling baby","crossing river","crying","curling hair","cutting nails","cutting pineapple","cutting watermelon","dancing ballet","dancing charleston","dancing gangnam style","dancing macarena","deadlifting","decorating the christmas tree","digging","dining","disc golfing","diving cliff","dodgeball","doing aerobics","doing laundry","doing nails","drawing","dribbling basketball","drinking","drinking beer","drinking shots","driving car","driving tractor","drop kicking","drumming fingers","dunking basketball","dying hair","eating burger","eating cake","eating carrots","eating chips","eating doughnuts","eating hotdog","eating ice cream","eating spaghetti","eating watermelon","egg hunting","exercising arm","exercising with an exercise ball","extinguishing fire","faceplanting","feeding birds","feeding fish","feeding goats","filling eyebrows","finger snapping","fixing hair","flipping pancake","flying kite","folding clothes","folding napkins","folding paper","front raises","frying vegetables","garbage collecting","gargling","getting a haircut","getting a tattoo","giving or receiving award","golf chipping","golf driving","golf putting","grinding meat","grooming dog","grooming horse","gymnastics tumbling","hammer throw","headbanging","headbutting","high jump","high kick","hitting baseball","hockey stop","holding snake","hopscotch","hoverboarding","hugging","hula hooping","hurdling","hurling (sport)","ice climbing","ice fishing","ice skating","ironing","javelin throw","jetskiing","jogging","juggling balls","juggling fire","juggling soccer ball","jumping into pool","jumpstyle dancing","kicking field goal","kicking soccer ball","kissing","kitesurfing","knitting","krumping","laughing","laying bricks","long jump","lunge","making a cake","making a sandwich","making bed","making jewelry","making pizza","making snowman","making sushi","making tea","marching","massaging back","massaging feet","massaging legs","massaging person's head","milking cow","mopping floor","motorcycling","moving furniture","mowing lawn","news anchoring","opening bottle","opening present","paragliding","parasailing","parkour","passing American football (in game)","passing American football (not in game)","peeling apples","peeling potatoes","petting animal (not cat)","petting cat","picking fruit","planting trees","plastering","playing accordion","playing badminton","playing bagpipes","playing basketball","playing bass guitar","playing cards","playing cello","playing chess","playing clarinet","playing controller","playing cricket","playing cymbals","playing didgeridoo","playing drums","playing flute","playing guitar","playing harmonica","playing harp","playing ice hockey","playing keyboard","playing kickball","playing monopoly","playing organ","playing paintball","playing piano","playing poker","playing recorder","playing saxophone","playing squash or racquetball","playing tennis","playing trombone","playing trumpet","playing ukulele","playing violin","playing volleyball","playing xylophone","pole vault","presenting weather forecast","pull ups","pumping fist","pumping gas","punching bag","punching person (boxing)","push up","pushing car","pushing cart","pushing wheelchair","reading book","reading newspaper","recording music","riding a bike","riding camel","riding elephant","riding mechanical bull","riding mountain bike","riding mule","riding or walking with horse","riding scooter","riding unicycle","ripping paper","robot dancing","rock climbing","rock scissors paper","roller skating","running on treadmill","sailing","salsa dancing","sanding floor","scrambling eggs","scuba diving","setting table","shaking hands","shaking head","sharpening knives","sharpening pencil","shaving head","shaving legs","shearing sheep","shining shoes","shooting basketball","shooting goal (soccer)","shot put","shoveling snow","shredding paper","shuffling cards","side kick","sign language interpreting","singing","situp","skateboarding","ski jumping","skiing (not slalom or crosscountry)","skiing crosscountry","skiing slalom","skipping rope","skydiving","slacklining","slapping","sled dog racing","smoking","smoking hookah","snatch weight lifting","sneezing","sniffing","snorkeling","snowboarding","snowkiting","snowmobiling","somersaulting","spinning poi","spray painting","spraying","springboard diving","squat","sticking tongue out","stomping grapes","stretching arm","stretching leg","strumming guitar","surfing crowd","surfing water","sweeping floor","swimming backstroke","swimming breast stroke","swimming butterfly stroke","swing dancing","swinging legs","swinging on something","sword fighting","tai chi","taking a shower","tango dancing","tap dancing","tapping guitar","tapping pen","tasting beer","tasting food","testifying","texting","throwing axe","throwing ball","throwing discus","tickling","tobogganing","tossing coin","tossing salad","training dog","trapezing","trimming or shaving beard","trimming trees","triple jump","tying bow tie","tying knot (not on a tie)","tying tie","unboxing","unloading truck","using computer","using remote controller (not gaming)","using segway","vault","waiting in line","walking the dog","washing dishes","washing feet","washing hair","washing hands","water skiing","water sliding","watering plants","waxing back","waxing chest","waxing eyebrows","waxing legs","weaving basket","welding","whistling","windsurfing","wrapping present","wrestling","writing","yawning","yoga","zumba"]
    #
    # labels = [line.strip() for line in labels]
    # print(output[0][0])
    # print('Total labels: {}'.format(len(labels)))
    #
    # print('Position of maximum probability: {}'.format(output.argmax()))
    # print('Maximum probability: {:.5f}'.format(max(output[0])))
    # print('Corresponding label: {}'.format(labels[output.argmax()]))
    #
    # # sort top five predictions from softmax output
    # top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    # print('\nTop 5 probabilities and labels:')
    # _ = [print('{:.5f} {}'.format(output[0][i], labels[i])) for i in top_inds]
    # print(output)
    # Show the model architecture
    new_model = tf.keras.models.load_model('../models/pretrained/rplus1')

    new_model.trainable = False
    print(new_model.signatures["serving_default"])
    predictor = new_model.signatures["serving_default"]
    x = np.zeros((1, 3, 112, 112, 8), dtype=np.float32)
    labeling = predictor(tf.constant(x, dtype=tf.float32))
    print(labeling['635'].shape)
