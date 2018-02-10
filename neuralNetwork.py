from __future__ import print_function
#import potrebnih biblioteka
#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab

import pickle

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 50, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    #Transformisati selektovani region na sliku dimenzija 28x28
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roi_from_video(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    regions_original = []
    brojevi = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if h > 17 and h < 30 and w < 30 and area>80:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), [x, y, w, h]])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)

        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions, regions_array


def select_roi(image_orig, image_bin):
    # Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
    #     Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
    #     Za oznacavanje regiona koristiti metodu cv2.boundingRect(contour).
    #     Kao povratnu vrednost vratiti originalnu sliku na kojoj su obelezeni regioni
    #     i niz slika koje predstavljaju regione sortirane po rastucoj vrednosti x ose
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    brojevi = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if h > 12 and w > 2 and h < 65 and w < 65 and area > 50:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if y<100:
                brojevi.append(0)
            elif y<200:
                brojevi.append(1)
            elif y<300:
                brojevi.append(2)
            elif y<400:
                brojevi.append(3)
            elif y<500:
                brojevi.append(4)
            elif y<600:
                brojevi.append(5)
            elif y<700:
                brojevi.append(6)
            elif y<800:
                brojevi.append(7)
            elif y<900:
                brojevi.append(8)
            elif y<1000:
                brojevi.append(9)

    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    brojevi.sort()

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, brojevi

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    #  Elementi matrice image su vrednosti 0 ili 255.
    #     Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    return image/255

def matrix_to_vector(image):
    #Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa
    return image.flatten()


def prepare_for_ann(regions):
    # Regioni su matrice dimenzija 28x28 ciji su elementi vrednosti 0 ili 255.
    #     Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann

def convert_output(alphabet, brojevi):
    #     Konvertovati alfabet u niz pogodan za obucavanje NM,
    #     odnosno niz ciji su svi elementi 0 osim elementa ciji je
    #     indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
    #     Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
    #     za drugi [0,1,0,0,0,0,0,0,0,0] itd..

    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(10)
        output[brojevi[index]] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():
    # Implementacija vestacke neuronske mreze sa 784 neurona na uloznom sloju,
    #     128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.

    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    #Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    with open('NN3.tw', 'wb') as output: pickle.dump(ann, output, pickle.HIGHEST_PROTOCOL)
    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    #pronaci i vratiti indeks neurona koji je najvise pobudjen
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    #za svaki rezultat pronaci indeks pobednickog
        # regiona koji ujedno predstavlja i indeks u alfabetu.
        # Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def getNumberFromRegion(region):

    test_inputs = []
    test_inputs.append(matrix_to_vector(scale_to_range(region)))
    result = ann.predict(np.array(test_inputs, np.float32))
    #print(result)
    return display_result(result, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[0]

with open('NN3.tw', 'rb') as input: ann = pickle.load(input)

image_color = load_image('images/numbers.png')
img = invert(image_bin(image_gray(image_color)))
img_bin = erode(dilate(img))
selected_regions, regions, brojevi = select_roi(image_color.copy(), img)
# # display_image(selected_regions)
# # plt.imshow(selected_regions)
# # plt.show()
#
# inputs = prepare_for_ann(regions)
# outputs = convert_output(regions, brojevi)
# ann = create_ann()
# ann = train_ann(ann, inputs, outputs)
