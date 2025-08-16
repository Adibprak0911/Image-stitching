# Image_stitching_panorama.py
import cv2
import numpy as np
import os
from Image_stitching_simple import (
    compute_sift_keypoints_and_descriptors, match_sift_descriptors, draw_matches, compute_and_warp, show_windows_sequentially
)

root = os.getcwd()
folder = os.path.join(root, 'image_stitching/images')
images = []

# Load images in order
idx = 1
while True:
    filename = os.path.join(folder, f'image{idx}.jpg')
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is None:
            break
        images.append(img)
        idx += 1
    else:
        break

if len(images) < 2:
    print("Need at least two images for stitching.")
    exit()

panorama = images[0]
for i in range(1, len(images)):
    img1 = panorama
    img2 = images[i]

    # SIFT keypoints and descriptors
    kp1, des1, img1_kp = compute_sift_keypoints_and_descriptors(img1)
    kp2, des2, img2_kp = compute_sift_keypoints_and_descriptors(img2)

    # Match descriptors
    good_matches = match_sift_descriptors(des1, des2)
    img_matches = draw_matches(img1, kp1, img2, kp2, good_matches)

    # Try to stitch and update panorama
    new_panorama = compute_and_warp(img1, kp1, img2, kp2, good_matches)
    if new_panorama is None:
        print(f"Stitching failed at image {i+1}.")
        break
    panorama = new_panorama

    # Show visualizations sequentially
    windows = [
        (f'SIFT Keypoints: Image {i}', img1_kp),
        (f'SIFT Keypoints: Image {i+1}', img2_kp),
        (f'SIFT Matches {i} to {i+1}', img_matches),
        (f'Stitched Panorama Step {i+1}', panorama)
    ]
    show_windows_sequentially(windows, delay_ms=5000)

# Show final panorama
if panorama is not None:
    cv2.imshow('Final Stitched Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
