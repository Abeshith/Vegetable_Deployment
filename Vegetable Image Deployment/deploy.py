import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Vegetable Image Classification")

# Load the model
model = load_model('model.h5')

# Define class labels
class_labels = {
    0: 'Bean',
    1: 'Bitter gourd',
    2: 'Bottle gourd',
    3: 'Brinjal',
    4: 'Broccoli',
    5: 'Cabbage',
    6: 'Capsicum',
    7: 'Carrot',
    8: 'Cauliflower',
    9: 'Cucumber',
    10: 'Papaya',
    11: 'Potato',
    12: 'Pumpkin',
    13: 'Radish',
    14: 'Tomato'
}

# Define vegetable features
vegetable_features = {
    'Bean': "Beans are highly nutritious and versatile. They are an excellent source of plant-based protein, which makes them a great addition to vegetarian diets. Beans are rich in fiber, which supports healthy digestion and can help manage blood sugar levels. They also contain essential vitamins and minerals, such as folate, iron, and magnesium. Regular consumption of beans has been linked to improved heart health, lower cholesterol levels, and better weight management. They can be used in a variety of dishes, from soups and stews to salads and casseroles.",
    'Bitter gourd': "Bitter gourd, also known as bitter melon, is a tropical vegetable known for its sharp, bitter taste. It's packed with essential nutrients like vitamins C and A, which contribute to a strong immune system and healthy skin. Bitter gourd is renowned for its potential blood-sugar-lowering effects, making it beneficial for individuals with diabetes. It also contains powerful antioxidants and anti-inflammatory compounds that can support overall health. This vegetable can be stir-fried, added to soups, or even consumed as a juice for its health benefits.",
    'Bottle gourd': "Bottle gourd, or lauki, is a low-calorie vegetable with a mild taste and high water content, making it an excellent choice for hydration. It is rich in dietary fiber, which aids in digestion and promotes satiety. Bottle gourd is known for its cooling properties, which can help soothe the digestive system and reduce acidity. It also provides essential vitamins and minerals, including vitamin C, potassium, and calcium. This vegetable can be cooked in various ways, including stews, soups, and curries, and is also used in traditional remedies for its health benefits.",
    'Brinjal': "Brinjal, commonly known as eggplant, is a nutrient-dense vegetable with a unique flavor and texture. It is an excellent source of dietary fiber, which supports digestive health and helps maintain healthy cholesterol levels. Brinjal is rich in antioxidants, including nasunin, which is found in its dark purple skin and helps protect cells from damage. It also provides essential vitamins and minerals such as vitamin C, vitamin K, and manganese. This versatile vegetable can be grilled, roasted, stuffed, or used in a variety of dishes from different cuisines.",
    'Broccoli': "Broccoli is a cruciferous vegetable renowned for its health benefits. It is exceptionally high in vitamins K and C, which play crucial roles in bone health and immune function. Broccoli is rich in antioxidants, including sulforaphane, which has been shown to have cancer-fighting properties. It also provides dietary fiber, which supports digestive health and helps maintain a healthy weight. This vegetable can be enjoyed raw, steamed, roasted, or added to a variety of dishes, including salads, stir-fries, and soups.",
    'Cabbage': "Cabbage is a leafy green vegetable that is both nutritious and versatile. It is high in vitamins C and K, which are important for immune health and blood clotting. Cabbage is also rich in fiber, which aids in digestion and helps maintain healthy cholesterol levels. Additionally, it contains antioxidants and anti-inflammatory compounds that can support overall health. Cabbage can be eaten raw in salads, fermented as sauerkraut, or cooked in soups and stews, making it a staple in many cuisines around the world.",
    'Capsicum': "Capsicum, commonly known as bell pepper, is a vibrant vegetable rich in essential nutrients. It is an excellent source of vitamins A and C, which support immune health, skin health, and vision. Capsicum is also high in antioxidants, including carotenoids, which can help protect against oxidative stress and inflammation. This vegetable comes in various colors, each with slightly different nutritional profiles and flavors. Capsicum can be eaten raw in salads, roasted, grilled, or added to a variety of dishes to enhance flavor and nutrition.",
    'Carrot': "Carrots are a root vegetable known for their vibrant orange color and sweet flavor. They are an excellent source of beta-carotene, which the body converts into vitamin A, promoting good vision and supporting immune health. Carrots are also rich in dietary fiber, which aids in digestion and helps maintain a healthy weight. Additionally, they contain antioxidants such as lutein and zeaxanthin, which can help protect against age-related macular degeneration. Carrots can be eaten raw, cooked, or juiced, and are a versatile ingredient in many recipes.",
    'Cauliflower': "Cauliflower is a cruciferous vegetable with a mild flavor and versatile uses in cooking. It is high in fiber, which supports digestive health and helps regulate blood sugar levels. Cauliflower is also rich in vitamins B and C, which are important for energy metabolism and immune function. Additionally, it contains antioxidants like sulforaphane that have been shown to have cancer-fighting properties. This vegetable can be enjoyed raw, steamed, roasted, or used as a low-carb substitute in dishes like rice and mashed potatoes.",
    'Cucumber': "Cucumbers are refreshing, hydrating vegetables with a high water content, making them ideal for maintaining hydration. They are low in calories and provide a good source of vitamins K and C, which support bone health and immune function. Cucumbers also contain antioxidants, such as beta-carotene and flavonoids, which help protect cells from damage. They are often eaten raw in salads or as a snack, and can also be pickled or used in beverages for added flavor and nutrition.",
    'Papaya': "Papaya is a tropical fruit known for its sweet flavor and rich nutritional profile. It is an excellent source of vitamins C and A, which support immune health and skin health. Papaya also contains the enzyme papain, which aids in digestion by breaking down proteins. Additionally, it provides fiber, which helps regulate bowel movements and supports digestive health. This fruit can be eaten fresh, blended into smoothies, or used in desserts and salads for its unique taste and health benefits.",
    'Potato': "Potatoes are a staple vegetable that provides a good source of carbohydrates, which are essential for energy. They are also rich in vitamin C, which supports immune health, and potassium, which helps regulate blood pressure. Potatoes contain dietary fiber, which aids in digestion and helps maintain a healthy weight. They can be prepared in various ways, including boiling, baking, roasting, or frying, and are a versatile ingredient in many dishes. Potatoes also offer essential nutrients and antioxidants that contribute to overall health.",
    'Pumpkin': "Pumpkins are nutrient-dense vegetables with a bright orange color, thanks to their high content of beta-carotene. This antioxidant is converted into vitamin A, which supports vision and immune health. Pumpkins are also rich in fiber, which aids in digestion and helps maintain a healthy weight. Additionally, they contain vitamins C and E, which have antioxidant properties. Pumpkins can be used in a variety of recipes, including soups, pies, and roasted dishes, and are a versatile ingredient for both sweet and savory dishes.",
    'Radish': "Radishes are crunchy root vegetables known for their peppery flavor. They are low in calories and high in dietary fiber, which supports digestion and helps regulate blood sugar levels. Radishes are also rich in vitamins C and B6, which are important for immune function and metabolism. They contain antioxidants such as anthocyanins and glucosinolates, which have been shown to have anti-inflammatory and detoxifying properties. Radishes can be eaten raw in salads, pickled, or added to soups and stir-fries for added crunch and nutrition.",
    'Tomato': "Tomatoes are a popular fruit with a tangy flavor and a rich nutritional profile. They are an excellent source of lycopene, an antioxidant that supports heart health and protects against certain types of cancer. Tomatoes also provide vitamins C and K, which are important for immune function and bone health. They contain dietary fiber, which aids in digestion and helps regulate blood sugar levels. Tomatoes can be eaten raw in salads, cooked in sauces, soups, or stews, and are a staple ingredient in many cuisines worldwide."
}

# Create a navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  # Required
        options=["Home", "Upload Image", "About"],  # Required
        icons=["house", "cloud-upload", "info-circle"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
        orientation="vertical",  # Optional
    )

# Display content based on selected menu option
if selected == "Home":
    st.title("Welcome to Vegetable Classification")
    st.write("Upload an image of a vegetable and the model will predict its class.")

elif selected == "Upload Image":
    st.title("Upload an Image")
    st.write("Upload your vegetable image here.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=False, width=200)  # Fixed size
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        class_name = class_labels[predicted_class[0]]

        st.write(f"**Prediction:** {class_name}")

        # Display features of the predicted vegetable
        st.write(f"**Features of {class_name}:** {vegetable_features[class_name]}")

elif selected == "About":
    st.title("About This App")
    st.write("This app classifies vegetables using a pre-trained model.")
    st.write("Upload an image of a vegetable, and the model will predict its class along with detailed features of the vegetable.")
