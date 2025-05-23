import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Optional


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image by converting it to grayscale and applying Gaussian blur.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray is None or gray.size == 0:
        raise ValueError("Failed to convert image to grayscale")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def save_image(img: np.ndarray, path: str, output_dir: str = 'results/image_classified/normal_img') -> None:
    """
    Save the processed image to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, path)
    if len(img.shape) == 2:
        plt.imsave(full_path, img, cmap='gray')
    else:
        plt.imsave(full_path, img)


def kmeans_cluster(img: np.ndarray, k: int = 2, attempts: int = 10) -> np.ndarray:
    """
    Perform K-means clustering to segment the input image into k clusters.
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image")
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(img.shape)


def detect_edges(img: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection to the input image.
    """
    if len(img.shape) not in (2, 3):
        raise ValueError("Input image must be grayscale or BGR")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, low_threshold, high_threshold)


def get_bounding_boxes(
    img: np.ndarray, 
    epsilon_factor: float = 0.02
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], List[np.ndarray]]:
    """
    Detect contours and bounding boxes in the input image.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    polygons = []
    for cnt in contours:
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        boxes.append((x, y, w, h))
        polygons.append(approx)
    return boxes, contours, polygons


def draw_contours(
    img: np.ndarray, 
    boxes: List[Tuple[int, int, int, int]], 
    color: Tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on the input image.
    """
    result = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def crop_to_largest_box(
    img: np.ndarray, 
    boxes: List[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop the input image to the largest bounding box.
    """
    if not boxes:
        return img.copy(), img.copy()
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    cropped = img[y:y + h, x:x + w]
    return cropped, cropped.copy()


def overlay_images(
    img1: np.ndarray, 
    img2: np.ndarray, 
    alpha: float = 0.5, 
    beta: float = 0.5, 
    gamma: float = 0
) -> np.ndarray:
    """
    Overlay two images with specified transparency.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)


def calculate_foot_size(
    cropped_img: np.ndarray, 
    px_to_cm_ratio: float = 0.026
) -> Tuple[float, float, float, float, float]:
    """
    Calculate the foot's dimensions in pixels and centimeters based on the input image.
    """
    px_height = cropped_img.shape[0]
    px_width = cropped_img.shape[1]
    foot_length = px_height * px_to_cm_ratio
    foot_width = px_width * px_to_cm_ratio
    return px_height, px_width, foot_length, foot_width, 0


def calculate_px_to_cm_ratio(a4_box: Tuple[int, int, int, int], assume_vertical: bool = True) -> float:
    """
    Calculate the pixel-to-centimeter ratio based on the A4 paper bounding box.
    """
    _, _, w, h = a4_box
    if assume_vertical:
        return 29.7 / h  # cm per pixel
    else:
        return 29.7 / w


# === Example Processing Pipeline (Usage) ===
# image = cv2.imread('your_image.jpg')
# blurred = preprocess(image)
# edges = detect_edges(blurred)
# boxes, contours, polygons = get_bounding_boxes(edges)
# a4_box = max(boxes, key=lambda b: b[2] * b[3])  # Assume largest box is A4
# px_to_cm = calculate_px_to_cm_ratio(a4_box, assume_vertical=True)
# cropped_img, _ = crop_to_largest_box(image, boxes)
# results = calculate_foot_size(cropped_img, px_to_cm)
# print("Foot size:", results)
