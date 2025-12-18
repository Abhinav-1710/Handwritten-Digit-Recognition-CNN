import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
CANVAS_SIZE = 280  # 280x280 pixels for the drawing area
PREDICTION_IMAGE_SIZE = 28 # MNIST standard size
MODEL_PATH = os.path.join(os.getcwd(), 'model_files', 'cnn_mnist_model.h5')
CANVAS_SCALE = CANVAS_SIZE / PREDICTION_IMAGE_SIZE # Scale factor (280/28 = 10)

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Recognizer")

        # Load the trained model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model successfully loaded from: {MODEL_PATH}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}\nEnsure train_mnist.py was run successfully.")
            master.destroy()
            return
        
        # Initialize image to draw on (White background for PIL)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0) # 'L' for grayscale, black background (0)
        self.draw = ImageDraw.Draw(self.image)

        # --- GUI Elements ---
        
        # 1. Canvas for Drawing
        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor="dot")
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint) # Left mouse button click and drag
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit) # On release of the mouse button

        # 2. Prediction Display
        self.prediction_label = tk.Label(master, text="Draw a digit (0-9)", font=("Helvetica", 24))
        self.prediction_label.pack(pady=5)

        # 3. Control Buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10)

        # Clear Button
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # Predict Button (Optional, but good backup)
        self.predict_button = tk.Button(self.button_frame, text="Predict Now", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        # Clear Tkinter canvas
        self.canvas.delete("all")
        # Reset PIL image to black
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Canvas Cleared. Draw a digit (0-9)")
        self.last_x, self.last_y = None, None

    def paint(self, event):
        # Coordinates of the drawing stroke
        x, y = event.x, event.y
        
        # Drawing color and size (White color, large stroke)
        paint_color = 'white'
        paint_width = 20 # Large brush size for drawing on 280x280 canvas
        
        if self.last_x and self.last_y:
            # Draw line on Tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill=paint_color, width=paint_width, capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw line on PIL image (for actual prediction)
            self.draw.line([self.last_x, self.last_y, x, y], 
                           fill=255, width=int(paint_width * 0.8), joint='round')

        self.last_x, self.last_y = x, y
        
    def predict_digit(self, event=None):
        if not self.last_x: # Prevent prediction if canvas is empty
            self.prediction_label.config(text="Draw something first!")
            return

        # 1. Resize Image: Downscale the 280x280 drawing to 28x28
        # We use Image.LANCZOS for high-quality downsampling.
        img_resized = self.image.resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # 2. Convert to NumPy Array
        # The image is inverted because MNIST expects white digit on a black background
        img_array = np.array(img_resized) 

        # 3. Preprocessing (Same as training data): Normalization and Reshaping
        img_processed = img_array.astype('float32') / 255.0
        img_processed = img_processed.reshape(1, PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE, 1)

        # 4. Predict
        prediction = self.model.predict(img_processed)
        predicted_class = np.argmax(prediction)
        
        # Get the probability of the predicted class
        confidence = prediction[0][predicted_class] * 100

        # 5. Display Result
        self.prediction_label.config(
            text=f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)",
            font=("Helvetica", 36, "bold")
        )

# Main Application Run
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    # Prevent resizing the window
    root.resizable(False, False)
    root.mainloop()