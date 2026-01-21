import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os


def train_and_plot(model, train_input, train_target, val_data=None, 
                   epochs=20, callbacks=None):
   
    print("\n--- 훈련 시작 ---")
    history = model.fit(train_input, train_target, epochs=epochs, verbose=0,
                        validation_data=val_data, callbacks=callbacks)
    print("--- 훈련 완료 ---")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'])
    if val_data: 
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'val'])
    else:
        plt.legend(['train'])
            
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.show()
    
   
    if 'accuracy' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'])
        if val_data:
            plt.plot(history.history['val_accuracy'])
            plt.legend(['train', 'val'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy Curve')
        plt.show()
        
    return history

def evaluate_model(model, val_input, val_target):
    """
    모델 평가 결과를 출력하는 함수
    """
    print("\n[모델 평가]")
    results = model.evaluate(val_input, val_target)
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
    print("-" * 30)


(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28))) 
    model.add(keras.layers.Dense(100, activation='relu')) 
    if a_layer:
        model.add(a_layer) 
    model.add(keras.layers.Dense(10, activation='softmax')) 
    return model



print("1. 기본 모델 훈련")
model = model_fn()
model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_and_plot(model, train_scaled, train_target, epochs=5)


# 2. 기본 모델 (Epoch 20으로 증가)
print("2. Epoch 증가 모델")
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_and_plot(model, train_scaled, train_target, epochs=20)



print("3. 검증 데이터 추가")
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_and_plot(model, train_scaled, train_target, 
               val_data=(val_scaled, val_target), epochs=20)



print("4. Adam 옵티마이저 적용")
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_and_plot(model, train_scaled, train_target, 
               val_data=(val_scaled, val_target), epochs=20)



print("5. 드롭아웃 적용")

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_and_plot(model, train_scaled, train_target, 
               val_data=(val_scaled, val_target), epochs=20)



print("6. 모델 저장 및 불러오기")
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=10, verbose=0, 
          validation_data=(val_scaled, val_target))

model.save_weights('model-weights.weights.h5')
model.save('model-whole.h5')

print("현재 파일 목록:", os.listdir())


model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.weights.h5')


val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print("예측 정확도 평균:", np.mean(val_labels == val_target))


model = keras.models.load_model('model-whole.h5')
evaluate_model(model, val_scaled, val_target)

print("7. 체크포인트 콜백 적용")
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)

train_and_plot(model, train_scaled, train_target, 
               val_data=(val_scaled, val_target), 
               epochs=20, callbacks=[checkpoint_cb])

model = keras.models.load_model('best-model.h5')
evaluate_model(model, val_scaled, val_target)


print("8. 조기 종료(Early Stopping) 콜백 적용")
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_weight=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weight=True)

history = train_and_plot(model, train_scaled, train_target, 
                         val_data=(val_scaled, val_target), 
                         epochs=20, callbacks=[checkpoint_cb, early_stopping_cb])

model = keras.models.load_model('best-model.h5')
evaluate_model(model, val_scaled, val_target)