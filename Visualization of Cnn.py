from tensorflow import keras 
model=keras.models.load_model('best-cnn-model.keras')
print(model.layers)
conv= model.layers[0]
print(conv.weights[0].shape,conv.weights[1].shape)

conv_weights= conv.weights[0].numpy()
print(conv_weights.mean(),conv_weights.std())

import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
#check for cnn weights
fig,axs = plt.subplots(2,16,figsize=(15,2))
for i in range(2):
  for j in range(16):
    axs[i,j].imshow(conv_weights[:,:,0,i*16 +j],vmin=-0.5,vmax=0.5)
    axs[i,j].axis('off')
plt.show()

#compare to none training cnn model
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32,kernel_size=3,activation=\
                                          'relu',padding='same',input_shape=(28,28,1)))
no_training_conv= no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

no_training_weights=no_training_conv.weights[0].numpy()
print(no_training_weights.mean(),no_training_weights.std())

plt.hist(no_training_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig,axs=plt.subplots(2,16,figsize=(15,2))
for i in range(2):
  for j in range(16):
    axs[i,j].imshow(no_training_weights[:,:,0,i*16+j], vmin=-0.5,
                    vmax=0.5)
    axs[i,j].axis('off')
plt.show()

#Functional api

model.build(input_shape=(None,28,28,1))
conv_acti= keras.Model(model.inputs[0],model.layers[0].output)

(train_input,train_target),(test_input,test_target)=\
keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0],cmap='gray_r')
plt.show()
inputs = train_input.reshape(-1,28,28,1)/ 255.0
feature_maps=conv_acti.predict(inputs)

print(feature_maps.shape)

fig,axs=plt.subplots(4,8,figsize=(15,8))
for i in range(4):
  for j in range(8):
    axs[i,j].imshow(feature_maps[0,:,:,i*8 + j])
    axs[i,j].axis('off')
plt.show()

conv2_acti= keras.Model(model.inputs[0],model.layers[1].output)

inputs =train_input[0:1].reshape(-1,28,28,1)/255.0
feature_maps= conv2_acti.predict(inputs)
print(feature_maps.shape)

fig,axs=plt.subplots(4,8,figsize=(12,12))
for i in range(4):
  for j in range(8):
    axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
    axs[i,j].axis('off')
plt.show()

