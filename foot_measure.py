import streamlit as st
import pandas as pd
import numpy as np
import skimage
from foot_estimate import *
import os
import cv2
import sqlite3
from datetime import datetime
from PIL import Image
import uuid
from classifier import ImageClassifier
import io
import base64
from manual_adjust import manual_a4_adjustment, manual_foot_adjustment

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('foot_measurement.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY,
                  name TEXT,
                  password TEXT,
                  email TEXT,
                  role TEXT DEFAULT 'user')''')

    c.execute('''CREATE TABLE IF NOT EXISTS measurements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  timestamp DATETIME,
                  foot_length REAL,
                  size_vn TEXT,
                  size_uk TEXT,
                  size_us TEXT,
                  size_eu TEXT)''')

    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                 ('admin', 'Admin', 'admin123', 'admin@example.com', 'admin'))

    conn.commit()
    conn.close()

def save_measurement(username, foot_length, size_vn, size_uk, size_us, size_eu):
    conn = sqlite3.connect('foot_measurement.db')
    c = conn.cursor()
    c.execute('''INSERT INTO measurements 
                 (username, timestamp, foot_length, size_vn, size_uk, size_us, size_eu)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (username, datetime.now(), foot_length, size_vn, size_uk, size_us, size_eu))
    conn.commit()
    conn.close()

def add_user(username, name, password, email, role='user'):
    conn = sqlite3.connect('foot_measurement.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                 (username, name, password, email, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

init_db()

# --- IMAGE ENHANCEMENT FUNCTION ---
def enhance_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

# --- CUSTOM image_to_url FUNCTION ---
def image_to_url(img: Image.Image):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- UI ---
st.title('Welcome To Project Foot Size Estimation!')
st.write("Upload an image to get your foot size measurement:")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    height_size = [
        15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
        20.5, 21.0, 21.6, 22.0, 22.4, 22.9, 23.3, 23.7, 24.0, 24.4,
        25.0, 25.7, 26.0, 26.4, 26.8, 27.1, 27.5, 27.9, 28.3, 28.8
    ]

    Size_UK_h = [str(round(8.5 - 0.5 * (24 - i), 1)).rstrip('0').rstrip('.') for i in range(len(height_size))]
    Size_EU_h = [str(round(float(uk) * 1.27 + 33, 1)).rstrip('0').rstrip('.') for uk in Size_UK_h]
    Size_US_h = [str(round(float(uk) + 1, 1)).rstrip('0').rstrip('.') for uk in Size_UK_h]
    Size_VN_h = Size_EU_h

    st.image(uploaded_file, caption='Input Image', use_container_width=True)

    try:
        og_img = skimage.io.imread(uploaded_file)
    except:
        st.error("Error reading the image. Please upload a valid image file.")
        st.stop()

    enhanced_img = enhance_image(og_img)
    st.image(enhanced_img, caption='Enhanced Image', use_container_width=True)

    test_img = cv2.cvtColor(enhanced_img.copy(), cv2.COLOR_RGB2BGR)
    test_img = cv2.resize(test_img, (128, 128))
    test_img = test_img / 255
    test_img = test_img.flatten()

    clf = ImageClassifier()
    predicted_class_num = clf.predict(test_img)
    predicted_class_name = clf.reverse_dict_label[predicted_class_num]

    class_options = list(clf.dict_label.keys())
    selected_class_name = st.selectbox("Predicted Class (you can change it if needed):", 
                                       class_options, 
                                       index=class_options.index(predicted_class_name))
    img_class = clf.dict_label[selected_class_name]
    st.info(f"[CLASS CONFIRMED] Proceeding with: **{selected_class_name}**")

    if st.button("Save Uploaded Image to Classified Folder"):
        save_dir = os.path.join("image_classified", selected_class_name)
        os.makedirs(save_dir, exist_ok=True)
        img = Image.open(uploaded_file)
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)
        st.success(f"Image saved to {save_path}")

    preprocess_img = preprocess(enhanced_img, img_class)
    st.image(preprocess_img, caption='Preprocessed Image', use_container_width=True)

    clustered_img = kMeans_cluster(preprocess_img, img_class)
    st.image(clustered_img, caption='KMeans Clustered Image', use_container_width=True)

    edge_detected_img = paperEdgeDetection(clustered_img)
    st.image(edge_detected_img, caption='Canny Edge Detected', use_container_width=True)

    boundRect, contours, contours_poly, img = getBoundingBox(edge_detected_img)

    a4_box = boundRect[0] if img_class == 0 else boundRect[1]

    # Show auto-detected A4 bounding box
    pdraw = drawCnt(a4_box, contours, contours_poly, img)
    st.image(pdraw, caption='[AUTO] A4 Paper Bounding Box (Auto-Detected)', use_container_width=True)

    # Checkbox to enable manual A4 box adjustment
    manual_adjust_a4 = st.checkbox("Manually adjust A4 bounding box")

    if manual_adjust_a4:
        manual_a4_box = manual_a4_adjustment(a4_box, contours, contours_poly, img)
    else:
        manual_a4_box = a4_box

    cropped_img, pcropped_img = cropOrig(manual_a4_box, clustered_img)
    st.image(cropped_img, caption='Cropped Image', use_container_width=True)

    new_img = overlayImage(cropped_img, pcropped_img)
    st.image(new_img, caption='Overlay Image', use_container_width=True)

    fedged = footEdgeDetection(new_img)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)

    fdraw = drawCnt(fboundRect[0], fcnt, fcntpoly, fimg)
    st.image(fdraw, caption='Foot Edge Detected (Auto)', use_container_width=True)

    # Checkbox to enable manual Foot box adjustment
    manual_adjust_foot = st.checkbox("Manually adjust Foot bounding box")

    if manual_adjust_foot:
        manual_foot_box = manual_foot_adjustment(fboundRect[0], fcnt, fcntpoly, fimg)
    else:
        manual_foot_box = fboundRect[0]

    ofs_w, ofs_h, fh_, fw_, ph, pw = calcFeetSize(pcropped_img, [manual_foot_box])
    foot_length = round(ofs_h, 3)
    st.write(f"[INFO] Foot's length (cm): {foot_length}")

    index_h = -1
    for i in range(len(height_size) - 1):
        if height_size[i] <= ofs_h <= height_size[i + 1]:
            index_h = i
            break

    if index_h != -1:
        size_vn = Size_VN_h[index_h]
        size_uk = Size_UK_h[index_h]
        size_us = Size_US_h[index_h]
        size_eu = Size_EU_h[index_h]
    else:
        size_vn = size_uk = size_us = size_eu = "Out of Range"

    st.write(f"[FOOT LENGTH SIZE] VN: {size_vn} | UK: {size_uk} | US: {size_us} | EU: {size_eu}")

    st.subheader("Would you like to create an account to save this measurement?")
    create_account = st.radio("Choose an option:",
                              ("Yes, I want to create an account", "No, I'll proceed without saving"))

    if create_account == "Yes, I want to create an account":
        with st.form("register_form"):
            st.write("Please fill in the details:")
            name = st.text_input("Full Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            email = st.text_input("Email")
            submitted = st.form_submit_button("Register and Save")

            if submitted:
                success = add_user(username, name, password, email)
                if success:
                    save_measurement(username, foot_length, size_vn, size_uk, size_us, size_eu)
                    st.success("Registration successful and measurement saved!")
                else:
                    st.error("Username already exists. Try another one.")
    else:
        st.info("Thank you for using the Foot Size Estimation app!")