import logging
from numbers import Real

import tensorflow as tf
from kapre.time_frequency import Melspectrogram
import numpy as np
import soundfile as sf
import resampy
from keras import Input
from tensorflow import function

TARGET_SR = 48000

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, Conv3D
from keras.optimizers import SGD
from keras.layers import *
from keras.models import Model


def prelu_channelwise(x, name=None):
    return PReLU(shared_axes=[1, 2], name=name)(x)


def bottleneck_IR(x, in_channel, depth, stride=1, prefix=None):
    if in_channel == depth:
        shortcut = MaxPooling2D((1, 1), stride)(x)
    else:
        shortcut = Conv2D(depth, 1, strides=stride, use_bias=False, name=prefix + ".shortcut_layer.0")(x)
        shortcut = BatchNormalization(epsilon=1e-5, name=prefix + ".shortcut_layer.1")(shortcut)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix + ".res_layer.0")(x)
    x = Conv2D(depth, 3, padding="same", use_bias=False, name=prefix + ".res_layer.1")(x)
    x = prelu_channelwise(x, name=prefix + ".res_layer.2")
    x = ZeroPadding2D(1)(x)
    x = Conv2D(depth, 3, strides=stride, use_bias=False, name=prefix + ".res_layer.3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix + ".res_layer.4")(x)

    out = Add()([x, shortcut])
    return out


def IR50(input_size=(112, 112, 3), weights_path=None, model_name="IR50"):
    inp = Input(input_size)

    # input_layer
    x = Conv2D(64, 3, padding="same", use_bias=False, name="input_layer.0")(inp)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="input_layer.1")(x)
    x = prelu_channelwise(x, name="input_layer.2")

    # body
    # IR_50: [3, 4, 14, 3]; IR_101: [3, 13, 30, 3]; IR_152: [3, 8, 36, 3]
    # blocks 0
    x = bottleneck_IR(x, 64, 64, 2, prefix=f"body.0")
    for i in range(1, 3):
        x = bottleneck_IR(x, 64, 64, prefix=f"body.{str(i)}")

    # blocks 1
    x = bottleneck_IR(x, 64, 128, 2, prefix=f"body.3")
    for i in range(4, 7):
        x = bottleneck_IR(x, 128, 128, prefix=f"body.{str(i)}")

    # blocks 2
    x = bottleneck_IR(x, 128, 256, 2, prefix=f"body.7")
    for i in range(8, 21):
        x = bottleneck_IR(x, 256, 256, prefix=f"body.{str(i)}")

    # blocks 2
    x = bottleneck_IR(x, 256, 512, 2, prefix=f"body.21")
    for i in range(22, 24):
        x = bottleneck_IR(x, 512, 512, prefix=f"body.{str(i)}")

        # output_layer
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="output_layer.0")(x)
    x = Dropout(0.5)(x, training=False)
    x = Permute((3, 1, 2))(x)
    x = Flatten()(x)
    x = Dense(512, name="output_layer.3")(x)
    out = BatchNormalization(momentum=0.9, epsilon=1e-5, name="output_layer.4")(x)

    model = Model(inp, out, name=model_name)
    if weights_path is not None:
        model.load_weights(weights_path)

    return model


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
    # # Show the model architecture
    # new_model = tf.keras.models.load_model('../../models/pretrained/rplus1')
    #
    # new_model.trainable = False
    # print(new_model.signatures["serving_default"])
    # predictor = new_model.signatures["serving_default"]
    # x = np.zeros((1, 3, 112, 112, 8), dtype=np.float32)
    # labeling = predictor(tf.constant(x, dtype=tf.float32))
    # print(labeling['635'].shape)
    new_model = tf.keras.models.load_model('../../models/pretrained/ir50_ms1m_keras.h5')
    new_model.summary()
    frames = np.zeros((2, 112, 112, 3))
    intermediate_layer_model = Model(inputs=new_model.input,
                                     outputs=new_model.get_layer('flatten_1').output)
    # input_tensor = tf.keras.layers.Input(shape=(112, 112, 3))
    # x = input_tensor
    # for layer in new_model.layers[:-1]:
    #     x = layer(x)
    #     if layer.name == "dropout_1":
    #         break
    #
    # model = tf.keras.Model(inputs=input_tensor, outputs=x)
    #
    # layer_output = model.predict(frames)

    print(intermediate_layer_model.predict(frames).shape)
