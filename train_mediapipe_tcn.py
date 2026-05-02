import tensorflow as tf
from tensorflow.keras import layers, models
def residual_block(x, dilation, filters):
conv = layers.Conv1D(filters, 3, padding='causal', 
dilation_rate=dilation)(x)
conv = layers.BatchNormalization()(conv)
conv = layers.ReLU()(conv)
conv = layers.Conv1D(filters, 3, padding='causal', 
dilation_rate=dilation)(conv)
conv = layers.BatchNormalization()(conv)
conv = layers.ReLU()(conv)
if x.shape[-1] != filters:
x = layers.Conv1D(filters, 1)(x)
return layers.Add()([x, conv])
# Architecture
inputs = layers.Input(shape=(75, 40))
x = layers.Conv1D(64, 1)(inputs)
for i in range(4):  # Dilatation 1, 2, 4, 8
x = residual_block(x, 2**i, 64)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(20, activation='softmax')(x)
model = models.Model(inputs, outputs)
model.compile(optimizer='adam', 
loss='sparse_categorical_crossentropy', 
metrics=['accuracy'])