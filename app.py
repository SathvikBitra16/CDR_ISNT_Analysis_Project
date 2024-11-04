# %%

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from skimage import measure
from skimage import measure
from scipy.spatial import distance

app = Flask(__name__)
CORS(app)  # Enable CORS

# Define the path to the templates folder
TEMPLATES_FOLDER = os.path.join(os.path.dirname(__file__), 'templates')

# Define Dice coefficient
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define Jaccard index
def jaccard(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return (intersection + smooth) / (union + smooth)

# Define Dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Define BCE + Dice loss
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return 0.5 * bce + dl

# Load models for OD and OC segmentation
try:
    od_model = tf.keras.models.load_model(
        'models/OD_Bce_Dice.keras',
        custom_objects={'dice_coef': dice_coef, 'jaccard': jaccard, 'dice_loss': dice_loss, 'bce_dice_loss': bce_dice_loss}
    )
except Exception as e:
    print(f"Error occurred while loading the OD model: {e}")

try:
    oc_model = tf.keras.models.load_model(
        'models/OC_Bce_Dice.keras',
        custom_objects={'dice_coef': dice_coef, 'jaccard': jaccard, 'dice_loss': dice_loss, 'bce_dice_loss': bce_dice_loss}
    )
except Exception as e:
    print(f"Error occurred while loading the OC model: {e}")


# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe_to_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return clahe_image

# Function to calculate the minimum rim width between OD and OC contours
def calculate_min_rim_width(od_contours, oc_contours):
    min_distance = float('inf')
    min_od_point = None
    min_oc_point = None
    
    for od_contour in od_contours:
        for oc_contour in oc_contours:
            for od_point in od_contour:
                for oc_point in oc_contour:
                    dist = distance.euclidean(od_point, oc_point)
                    if dist < min_distance:
                        min_distance = dist
                        min_od_point = od_point
                        min_oc_point = oc_point

    return np.array(min_od_point, dtype=np.int32), np.array(min_oc_point, dtype=np.int32), min_distance

# Function to calculate OD diameter along the axis of minimum rim width
def calculate_od_diameter_along_rim_width_axis(od_points, min_od_point, min_oc_point):
    axis_vector = min_oc_point - min_od_point
    axis_vector_norm = axis_vector / np.linalg.norm(axis_vector)

    projected_points = [np.dot(p - min_od_point, axis_vector_norm) for p in od_points]
    min_projected = np.min(projected_points)
    max_projected = np.max(projected_points)

    min_od_axis_point = min_od_point + min_projected * axis_vector_norm
    max_od_axis_point = min_od_point + max_projected * axis_vector_norm

    od_diameter_along_axis = max_projected - min_projected

    return od_diameter_along_axis, min_od_axis_point, max_od_axis_point

# Function to calculate DDLS (Rim Width / OD Diameter Along Axis)
def calculate_ddls(min_rim_width, od_diameter_along_axis):
    if od_diameter_along_axis == 0:
        return float('inf')  # Avoid division by zero
    return min_rim_width / od_diameter_along_axis

# Function to calculate vertical and horizontal diameters for OD and OC
def calculate_diameters(contour_points):
    x_min = np.min(contour_points[:, 1])
    x_max = np.max(contour_points[:, 1])
    y_min = np.min(contour_points[:, 0])
    y_max = np.max(contour_points[:, 0])

    vertical_diameter = y_max - y_min
    horizontal_diameter = x_max - x_min

    return vertical_diameter, horizontal_diameter


def calculate_cdr(image_path):
    # Load the image and apply CLAHE for better contrast
    image = cv2.imread(image_path)
    if image is None:
        return {"result": "Image not found or cannot be opened"}

    # Apply CLAHE to the grayscale image generated from the RGB image
    grayscale_image = apply_clahe_to_image(image)

    # Resize the grayscale image for prediction
    grayscale_image_resized = cv2.resize(grayscale_image, (128, 128))
    rgb_image_resized = cv2.resize(image, (128, 128))

    # Normalize the grayscale image for model prediction
    grayscale_image_norm = grayscale_image_resized / 255.0
    grayscale_image_norm = np.expand_dims(grayscale_image_norm, axis=-1)  # Add channel dimension
    grayscale_image_norm = np.expand_dims(grayscale_image_norm, axis=0)  # Add batch dimension

    # Predict OD and OC masks
    od_prediction = od_model.predict(grayscale_image_norm)[0, :, :, 0]
    oc_prediction = oc_model.predict(grayscale_image_norm)[0, :, :, 0]

    # Binarize the predicted masks
    od_mask = (od_prediction > 0.5).astype(np.uint8)
    oc_mask = (oc_prediction > 0.5).astype(np.uint8)

    # Find contours of the predicted masks
    od_contours = measure.find_contours(od_mask, 0.5)
    oc_contours = measure.find_contours(oc_mask, 0.5)

    if not od_contours or not oc_contours:
        return {"result": "No OD or OC contours found in the image"}

    # Flatten contours
    od_points = np.vstack(od_contours).astype(np.int32)
    oc_points = np.vstack(oc_contours).astype(np.int32)

    # Calculate minimum rim width and the corresponding points
    min_od_point, min_oc_point, min_rim_width = calculate_min_rim_width(od_contours, oc_contours)

    # Calculate OD diameter along the same axis as the minimum rim width
    od_diameter_along_axis, min_od_axis_point, max_od_axis_point = calculate_od_diameter_along_rim_width_axis(od_points, min_od_point, min_oc_point)

    # Calculate DDLS
    ddls = calculate_ddls(min_rim_width, od_diameter_along_axis)

    # Calculate vertical and horizontal diameters for OD and OC
    od_vertical_diameter, od_horizontal_diameter = calculate_diameters(od_points)
    oc_vertical_diameter, oc_horizontal_diameter = calculate_diameters(oc_points)

    # Calculate VCDR and HCDR
    vcdr = oc_vertical_diameter / od_vertical_diameter
    hcdr = oc_horizontal_diameter / od_horizontal_diameter

    # Draw contours on grayscale and RGB images
    grayscale_contour_image = cv2.cvtColor(grayscale_image_resized, cv2.COLOR_GRAY2BGR)
    rgb_contour_image_resized = rgb_image_resized.copy()

    for contour in od_contours:
        contour = np.array(contour).astype(np.int32)
        cv2.drawContours(grayscale_contour_image, [contour], -1, (0, 255, 0), 1)
        cv2.drawContours(rgb_contour_image_resized, [contour], -1, (0, 255, 0), 1)

    for contour in oc_contours:
        contour = np.array(contour).astype(np.int32)
        cv2.drawContours(grayscale_contour_image, [contour], -1, (255, 0, 0), 1)
        cv2.drawContours(rgb_contour_image_resized, [contour], -1, (255, 0, 0), 1)

    # Draw minimum rim width line
    cv2.line(grayscale_contour_image, tuple(min_od_point), tuple(min_oc_point), (255, 255, 255), 2)
    cv2.line(rgb_contour_image_resized, tuple(min_od_point), tuple(min_oc_point), (255, 255, 255), 2)

    # Prepare results for response
    result = {
        "Disk Vertical Diameter": od_vertical_diameter,
        "Disk Horizontal Diameter": od_horizontal_diameter,
        "Cup Vertical Diameter": oc_vertical_diameter,
        "Cup Horizontal Diameter": oc_horizontal_diameter,
        "Vertical CDR": vcdr,
        "Horizontal CDR": hcdr,
        "Minimum Rim Width": min_rim_width,
        "OD Diameter Along Rim Width Axis": od_diameter_along_axis,
        "DDLS": ddls
    }

    # Optionally, save or return images with drawn contours
    # cv2.imwrite('output_grayscale.png', grayscale_contour_image)
    # cv2.imwrite('output_rgb.png', rgb_contour_image_resized)

    return result


@app.route('/')
def index():
    return send_from_directory(TEMPLATES_FOLDER, 'index.html')

@app.route('/register')
def register():
    return send_from_directory(TEMPLATES_FOLDER, 'register.html')

@app.route('/login')
def login():
    return send_from_directory(TEMPLATES_FOLDER, 'login.html')

@app.route('/image')
def image():
    return send_from_directory(TEMPLATES_FOLDER, 'image.html')

@app.route('/password')
def password():
    return send_from_directory(TEMPLATES_FOLDER, 'password.html')

@app.route('/about')
def about():
    pdf_path = 'static/pdfs/AboutUs.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/home')
def home():
    return send_from_directory(TEMPLATES_FOLDER, 'home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'result': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'No selected file'})

    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    # Perform CDR analysis
    cdr_result = calculate_cdr(file_path)


    # Remove the file after processing
    os.remove(file_path)

    # Combine results
    result = {
        "CDR Analysis": cdr_result
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
