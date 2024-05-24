from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Directory paths
train_dir = r'C:\Users\subin\PycharmProjects\Gaya_AA\aadhardata\train'
test_dir = r'C:\Users\subin\PycharmProjects\Gaya_AA\aadhardata\test'
valid_dir = r'C:\Users\subin\PycharmProjects\Gaya_AA\aadhardata\valid'

# Image size
image_size = (224, 224)
batch_size = 16  # Reduce batch size to reduce memory usage

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data generator for training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Data generator for validation and testing data
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define a simpler model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),  # Reduce the number of units
    tf.keras.layers.Dense(2, activation='softmax')  # Adjust the number of units based on your classification task
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size
)

# Save the trained model to a .h5 file
model.save('resnet_model_augmented.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
