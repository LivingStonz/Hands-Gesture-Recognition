import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation: Adding occlusions to simulate real-world scenarios
def augment_with_occlusions(images, occlusion_size=(50, 50)):
    augmented_images = []
    for img in images:
        img = tf.convert_to_tensor(img)
        h, w, _ = img.shape
        x = np.random.randint(0, w - occlusion_size[1])
        y = np.random.randint(0, h - occlusion_size[0])
        img = tf.tensor_scatter_nd_update(
            img,
            indices=tf.constant([[i, j] for i in range(y, y + occlusion_size[0]) 
                                 for j in range(x, x + occlusion_size[1])], dtype=tf.int64),
            updates=tf.zeros([occlusion_size[0] * occlusion_size[1], 3])
        )
        augmented_images.append(img.numpy())
    return np.array(augmented_images)

# Load the ChaLearn LAP dataset (example loader)
def load_chalearn_lap(data_dir, img_size=(128, 128)):
    """
    Placeholder for loading ChaLearn LAP dataset.
    Convert videos into frames and organize them into a directory structure:
    - data_dir/train/class_1/frame1.jpg
    - data_dir/train/class_2/frame2.jpg
    """
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    return train_data, val_data

# Define the CNN backbone
def build_cnn_backbone(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.GlobalAveragePooling2D()
    ])
    return model

# Define the Vision Transformer module
class VisionTransformer(layers.Layer):
    def __init__(self, num_patches, projection_dim, num_heads, transformer_layers):
        super(VisionTransformer, self).__init__()
        self.patch_embed = layers.Dense(projection_dim)
        self.transformer_blocks = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
            for _ in range(transformer_layers)
        ]
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs):
        x = self.patch_embed(inputs)
        for block in self.transformer_blocks:
            attn_output = block(x, x)
            x = self.layer_norm(x + attn_output)
        return x

# Build the hybrid model
def build_hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # CNN backbone
    cnn_backbone = build_cnn_backbone(input_shape)
    cnn_features = cnn_backbone(inputs)

    # Vision Transformer
    num_patches = cnn_features.shape[1]
    vit = VisionTransformer(num_patches=num_patches, projection_dim=128, num_heads=4, transformer_layers=2)
    vit_features = vit(cnn_features)

    # Classification head
    x = layers.Flatten()(vit_features)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

# Train the model
def train_model(data_dir, input_shape=(128, 128, 3), num_classes=20):
    train_data, val_data = load_chalearn_lap(data_dir, img_size=input_shape[:2])

    model = build_hybrid_model(input_shape, num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_data, validation_data=val_data, epochs=10, batch_size=32)
    model.save("hand_gesture_hybrid_model.h5")

# Main function
if __name__ == "__main__":
    data_dir = "/path/to/chalearn_lap_dataset"
    train_model(data_dir)

