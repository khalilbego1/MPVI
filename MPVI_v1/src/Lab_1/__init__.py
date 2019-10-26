import keras
import sys
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("-----Odgovori na pitanja za 3.1-----")

print("SKUP ZA TRENIRANJE:")
print("Broj slika: "+str(len(train_images)))
print("Rezolucija slika (pod pretpostavkom da su sve slike iste rezolucije): " + str(train_images[1].shape[1])+"x"+str(train_images[1].shape[1]))
print("Osnovne informacije na slikama su brojevi od 0 do 9, kao što se može vidjeti na ispisu labela:" + str(train_labels))
print("Skup zauzima: " + str(sys.getsizeof(train_images)) + " bajta")

print("SKUP ZA TESTIRANJE:")
print("Broj slika: "+str(len(test_images)))
print("Rezolucija slika (pod pretpostavkom da su sve slike iste rezolucije): " + str(test_images[1].shape[1])+"x"+str(test_images[1].shape[1]))
print("Osnovne informacije na slikama su brojevi od 0 do 9, kao što se može vidjeti na ispisu labela:" + str(test_labels))
print("Skup zauzima: " + str(sys.getsizeof(test_images)) + " bajta")

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32")/255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32")/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("-----Odgovori na pitanja za 3.2-----")

print("funkcija reshape u ovoj situaciji konvertuje 60 000 slika u niz jednodimenzionalnih slika, gdje je svaka slika jednodimenzionalni niz (a ne matrica, kao prije)")
print("funkcija astype konvertuje vrijednosti piksela svake od slika u floating point brojeve, nakon čega se dijele sa najvećom vrijednosti boje (255) da bi vrijednost bila između 0 i 1")
print("Jedan pixel sad zauzima " + str(sys.getsizeof(train_images[1][1])) +" bajta (32bitni floating point brojevi + overhead od PyObject_HEAD)")



print("-----Odgovori na pitanja za 3.3-----")

print("Epoha u fit funkciji predstavlja broj iteracija kroz čitav dataset")
print("Batch Size u fit funkciji predstavlja broj uzoraka po ažuriranju gradienta")
print("Proces treniranja u toku")
network.fit(train_images, train_labels, epochs=5, batch_size=128)
print("Proces treniranja završen")
print("Tačnost i vrijednosti funkcije gubitka ispisani iznad")

print("-----Odgovori na pitanja za 3.4-----")

test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Tačnost: " + str(test_acc) + " Vrijednost funkcije gubitka: " + str(test_loss))

