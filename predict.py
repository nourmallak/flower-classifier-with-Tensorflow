import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    return (image/255).numpy()

def predict(path, model, top_k):
    with Image.open(path) as img:
        img = np.asarray(img)
        processed_image = process_image(img)
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        probabilities, classes = tf.nn.top_k(prediction, k=top_k)
        return list(probabilities.numpy()[0]), list(classes.numpy()[0])

def main():
    parser = argparse.ArgumentParser(description='Predict the class of a flower image.')
    parser.add_argument('image_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str)
    
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    probabilities, classes = predict(args.image_path, model, args.top_k)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        classes = [class_names[str(class_ + 1)] for class_ in classes]

    for prob, class_n in zip(probabilities, classes):
        print('Class: {}, Probability: {}'.format(class_n, prob))

if __name__ == '__main__':
    main()

    


  