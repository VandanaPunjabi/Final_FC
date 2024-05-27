import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('FC.h5')
labels = {0: 'cabbage', 1: 'capsicum', 2: 'carrot', 3: 'cauliflower', 4:'chana masala', 5: 'chilli pepper', 6: 'corn', 
          7: 'cucumber',8: 'dhokla', 9: 'egg maggi', 10: 'eggplant', 11: 'garlic', 12: 'ginger', 13: 'grapes', 
          14: 'gulab jamun', 15: 'hakka noodles', 16: 'jalapeno', 17: 'kiwi', 18: 'korean maggi', 19: 'lemon', 20: 'lettuce',
          21:'maggi', 22: 'mango', 23:'masala maggi', 24:'masala oats', 25: 'masala upma',  26: 'onion', 
          27: 'orange', 28: 'paprika', 29: 'pear', 30: 'peas', 31: 'pineapple',
          32: 'ramen', 33: 'rolled oats', 34: 'upma', 35: 'vegetable soup'}

# Static calorie values per 100 grams
calorie_values = {
    'cabbage': 25, 'capsicum': 40, 'carrot': 41, 'cauliflower': 25, 'chana masala': 394.8, 'chilli pepper': 40,
    'corn': 86, 'cucumber': 16, 'dhokla': 355.5, 'egg maggi': 440, 'eggplant': 25, 'garlic': 149, 'ginger': 80, 
    'grapes': 69, 'gulab jamun': 410, 'hakka noodles': 371, 'jalapeno': 29, 'kiwi': 61, 'korean maggi': 321,
    'lemon': 29, 'lettuce': 15,  'mango': 60, 'masala maggi': 393, 'masala oats': 362,'maggi':310,
    'masala upma': 419.5, 'onion': 40, 'orange': 47, 'paprika': 282, 'pear': 57, 'peas': 81, 'pineapple': 50, 
    'ramen': 462, 'rolled oats': 345, 'upma': 393, 'vegetable soup': 375
}


dishes = [ 'Chilli Pepper', 'Grapes', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple','jalapeno','cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas','hakka noodles','masala maggi','masala oats','masala upma',
              'rolled oats','vegetable soup','chana masala','dhokla','egg maggi','garlic', 'korean maggi',
              'maggi','ramen', 'upma']


def get_calories(prediction):
    try:
        calories = calorie_values[prediction.lower()]
        return calories
    except KeyError:
        st.error("Calorie information not available for this item.")
        return None

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = labels[y]
    return res.capitalize()

def run():
    st.title(" fOOD CALORIES RECOGNITION SYSTEM")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './Test Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            if result:
                category = 'Dishe' if result.lower() in dishes else 'Dishes'
                st.info(f'**Category : {category}**')
                st.success(f"**Predicted : {result}**")
                cal = get_calories(result)
                if cal:
                    st.warning(f'**{cal} Kcal (per 100 grams)**')

run()
