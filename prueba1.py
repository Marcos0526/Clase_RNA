#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 21:31:53 2022

@author: marcosbautista
"""

import mnist_loader
import network
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

trainig_data = list(training_data)
test_data = list(test_data)

net= network.Network([784,30,10])

net.SGD( trainig_data, 30 , 10, 3.0, test_data = test_data)

archivo = open("red_prueba1.pk1",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#Leer el archivo

archivo_lectura = open("red_prueba1")
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD(training_data, 10, 50, 0.5, test_data=test_data)

archivo.open("red_prueba.pk1, 'wb' ")
pickle.dump(net,archivo)
archivo.close()
exit()

imagen = leer_imagen("disco.jpg")
print(net.feedforward(imagen))

