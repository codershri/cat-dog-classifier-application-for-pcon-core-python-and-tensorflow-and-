import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cats_and_dogs_classifier.h5')

# Initialize the main window
root = tk.Tk()
root.title("Cat and Dog Classifier")
root.geometry("500x600")

# Function to load and preprocess the image
def load_image():
    global img, img_display
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image with PIL
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize to match model's expected input
        img_array = np.array(img) / 255.0  # Normalize the image

        # Display the image in the GUI
        img_display = ImageTk.PhotoImage(img)
        panel.configure(image=img_display)
        panel.image = img_display

        # Predict the class
        prediction = predict_image(img_array)
        result_label.config(text=prediction)

# Function to predict the image class
def predict_image(image_array):
    # Expand dimensions to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    # Make prediction
    prediction = model.predict(image_array)
    # Interpret the prediction
    if prediction[0] > 0.5:
        return "It's a Dog!"
    else:
        return "It's a Cat!"

# GUI Components
btn = tk.Button(root, text="Load Image", command=load_image)
btn.pack(pady=20)

panel = tk.Label(root)  # Label to display the image
panel.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
