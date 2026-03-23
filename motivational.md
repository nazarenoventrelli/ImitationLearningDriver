El “modelo” sería una red neuronal que aprende a mapear:

pantalla → acción

o mejor:

secuencia corta de pantallas → acción

En tu caso, la acción puede ser algo simple como:

W
W+A
W+D
A
D
S
nada

Eso es un problema de aprendizaje supervisado.

Qué es el modelo

Pensalo como una función con parámetros:

f(imagen) = acción

Durante el entrenamiento, esa función ajusta millones de pesos internos para que, al ver una imagen parecida a las que vos manejaste, prediga la misma acción que tomaste vos.

Qué tipo de modelo usarías

Para esto, lo normal es una CNN o un backbone visual liviano.

Opciones razonables:

CNN chica hecha a mano
ResNet18 / ResNet34
EfficientNet liviana
MobileNet

Para arrancar, yo usaría algo tipo ResNet18 porque:

simple,
probada,
suficiente para dos pistas parecidas.
Qué entra al modelo
Opción simple

Un frame RGB.

Entrada:

imagen H x W x 3

Salida:

una clase de acción
Opción mejor

Varios frames seguidos.

Entrada:

3 o 4 frames concatenados
o frame actual + frame anterior

Eso ayuda a inferir movimiento.

Ejemplo:

con un solo frame, a veces no sabés si te estás acercando o alejando,
con varios frames, sí.
Qué sale del modelo
Variante A: clasificación

El modelo devuelve probabilidades para cada acción:

P(W), P(W+A), P(W+D), P(A), P(D), P(S), P(idle)

Y elegís la más alta.

Variante B: regresión

Devuelve algo como:

steer entre -1 y 1
throttle entre 0 y 1

Para teclado, yo arrancaría con clasificación. Es más simple.

Cómo se entrena
Paso 1: armás el dataset

Por ejemplo:

frame_000001.jpg → W
frame_000002.jpg → W+A
frame_000003.jpg → W+A

Eso lo convertís a números:

W = 0
W+A = 1
W+D = 2
A = 3
D = 4
S = 5
idle = 6
Paso 2: separás train / val / test

Por ejemplo:

70% train
20% val
10% test

Importante: mejor separar por vueltas completas, no por frames al azar.
Si mezclás frames de la misma vuelta en train y val, te engañás y parece que generaliza más de lo real.

Paso 3: preprocesás imágenes

Normalmente:

resize a algo como 224x224
normalización
quizá recorte del HUD si molesta

También podés usar data augmentation leve:

brillo
contraste
pequeño blur

Pero sin romper demasiado la geometría.

Paso 4: entrenamiento supervisado

Le mostrás al modelo:

imagen
etiqueta correcta

El modelo predice una acción.
Comparás su predicción con la acción real usando una loss, típicamente:

cross-entropy para clasificación

Y con backpropagation ajustás los pesos.

En términos concretos

En cada batch pasa esto:

entran 64 imágenes
el modelo predice 64 acciones
se calcula el error contra las acciones humanas
el optimizador corrige los pesos

Eso se repite por muchas épocas.

Ejemplo mental de entrenamiento

Supongamos que esta imagen corresponde a W+A.

El modelo predice:

W:    0.20
W+A:  0.35
W+D:  0.10
A:    0.15
D:    0.05
S:    0.05
idle: 0.10

Como la correcta era W+A, el entrenamiento ajusta los pesos para que la próxima vez suba la probabilidad de W+A.

Después de muchas iteraciones, aprende patrones visuales asociados a cada acción.

Pseudoflujo de entrenamiento
for epoch in range(num_epochs):
    for images, labels in train_loader:
        preds = model(images)
        loss = cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Eso es el núcleo.

Cómo sería el modelo en PyTorch, a alto nivel
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 acciones

Acá reutilizás una red preentrenada y cambiás la última capa para que prediga tus acciones.

Eso suele funcionar mucho mejor que empezar de cero.

Qué aprende internamente

No aprende “la pista” como humano simbólico. Aprende correlaciones visuales tipo:

si el target está desplazado a la izquierda, suele ser W+A
si está centrado, suele ser W
si el minimapa muestra un giro fuerte a la derecha, suele ser W+D

En capas profundas va armando representaciones útiles de:

bordes,
formas,
layout del HUD,
geometría del camino,
señales del minimapa.
Cómo se usa después de entrenarlo

Una vez entrenado:

capturás frame actual
lo preprocesás igual que en entrenamiento
lo pasás por el modelo
obtenés acción predicha
enviás esa tecla al juego

Loop de inferencia:

frame = capture()
x = preprocess(frame)
logits = model(x)
action = argmax(logits)
send_key(action)
Qué métricas mirar al entrenar

No sólo accuracy general. Miraría:

accuracy por clase
confusion matrix
recall en W+A y W+D
performance en validación por vuelta completa

Porque si el dataset está desbalanceado, puede sacar accuracy alta prediciendo siempre W.

Problema típico

La clase W suele dominar muchísimo.

Entonces necesitás:

balancear clases,
usar weighted loss,
o recolectar más curvas/correcciones.

Si no, el modelo aprende:
“casi siempre acelero recto”.

Qué arquitectura elegiría yo

Para un primer modelo serio:

entrada
frame actual
o 2–4 frames apilados
backbone
ResNet18
salida
5 a 7 clases de acción
loss
CrossEntropyLoss con pesos por clase
optimizador
AdamW

Eso ya alcanza para un muy buen baseline.

Mejoras después
1. Usar secuencia temporal

En vez de un solo frame:

stack de frames
o CNN + LSTM
2. Recortar regiones útiles

Podés darle:

frame principal
crop del minimapa

y fusionar ambos.

3. DAgger

Dejás que el modelo maneje, corregís, y agregás esos casos al dataset.

Eso mejora muchísimo.

Qué no es el modelo

No es:

YOLO
un detector de objetos
un planner clásico

Es una policy visual: una red que decide qué acción tomar mirando la pantalla.

Diferencia con YOLO
YOLO

Aprende:

dónde está un objeto

Entrada:

imagen

Salida:

bounding boxes
Behavioral Cloning

Aprende:

qué acción tomar

Entrada:

imagen

Salida:

acción
En tu caso concreto

Lo más razonable sería:

Modelo

Una red de clasificación de acciones

Input

Pantalla completa o pantalla + minimapa crop

Output

Una de estas acciones:

W
W+A
W+D
A
D
S
idle
Entrenamiento

Supervisado con frames y teclas grabadas

Resumen simple

El modelo sería una red neuronal que aprende a imitar tu conducción.

Se entrena así:

grabás frames + teclas
convertís cada tecla a una etiqueta
entrenás una CNN para predecir esa etiqueta desde la imagen
después usás esa red en tiempo real para decidir qué tecla tocar

Eso es Behavioral Cloning, una forma de Imitation Learning.

Puedo darte ahora un ejemplo más concreto de:
cómo se vería el dataset, el DataLoader y un script de entrenamiento mínimo en PyTorch.