# Proyecto de Inteligencia Artificial: Agente Taxi

Trabajo final de la asignatura de Inteligencia Artificial.

Este proyecto consiste en desarrollar y entrenar un agente inteligente capaz de interactuar con un entorno de juego mediante **aprendizaje por refuerzo**.

## Objetivo del proyecto

El agente debe aprender a desenvolverse en un entorno donde tiene que:

- Localizar y recoger a un pasajero situado en una casilla del mapa.
- Transportarlo hasta la ubicación de destino correspondiente.
- Aprender qué acciones son más adecuadas mediante la interacción con el entorno.
- Mejorar su comportamiento a partir de las recompensas obtenidas durante el entrenamiento.

## Aprendizaje por refuerzo

Requisito indispensable del trabajo usar Q-Learning

El aprendizaje por refuerzo se basa en la interacción entre un agente y un entorno.

En cada paso, el agente observa el estado del juego, selecciona una acción y recibe una recompensa o penalización según el resultado.

A través de sucesivas interacciones, aprende una estrategia que busca maximizar la recompensa acumulada.

## Requisitos

- Python 3.
- Gymnasium
- MoviePy
- Numpy
- Tqdm
- Pygame
- Matplotlib

## Versiones del proyecto

### El proyecto contempla tres aproximaciones al problema:

Entorno Gymnasium: implementación utilizando el entorno Taxi-v3 prediseñado. No tiene entorno visible, solo los datos de la simulacion.

Entorno con wrappers: implementación mediante wrappers para adaptar o modificar el entorno. Si tiene entorno visible.

Entorno propio: implementación utilizando un entorno desarrollado por los estudiantes. El usuario escoge si quiere ver el entorno mientras se ejecuta o no.

### Cada versión permite explorar distintas formas de representar el problema y entrenar al agente.

## Entornos

Esto es una captura de como se ve el entorno en su version con wrappers:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a48db408-db83-4398-8f92-0ea684e4b5d6" />

Esto es una captura de como se ve el entorno en su version con un entorno hecho por nosotros a mano:

<img width="507" height="552" alt="image" src="https://github.com/user-attachments/assets/ccffc17f-9a19-4238-80e7-f0b7596fc5ee" />

Un extra :) 

<img width="667" height="572" alt="image" src="https://github.com/user-attachments/assets/d9273bef-c186-402f-99d6-82c3ea5f72e0" />
