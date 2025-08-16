# image_stitching_simple.py
import cv2
import numpy as np

def compute_sift_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, descriptors, img_kp

def match_sift_descriptors(des1, des2, ratio_thresh=0.6, top_n=50):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    good = sorted(good, key=lambda x: x.distance)
    return good[:top_n] if len(good) > top_n else good

def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(0,255,0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def compute_and_warp(img1, kp1, img2, kp2, matches):
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
            corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
            corners_img1_trans = cv2.perspectiveTransform(corners_img1, H)
            all_corners = np.concatenate((corners_img1_trans, corners_img2), axis=0)
            xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            translation_dist = [-xmin, -ymin]
            H_translated = np.array([[1, 0, translation_dist[0]],
                                     [0, 1, translation_dist[1]],
                                     [0, 0, 1]])
            stitched = cv2.warpPerspective(img1, H_translated.dot(H), (xmax-xmin, ymax-ymin))
            stitched[translation_dist[1]:h2+translation_dist[1], translation_dist[0]:w2+translation_dist[0]] = img2
            return stitched
    return None

def show_windows_sequentially(windows, delay_ms=5000):
    """
    Show (window_name, image) pairs for delay_ms each, one at a time.
    """
    for win_name, img in windows:
        cv2.imshow(win_name, img)
        cv2.waitKey(delay_ms)
        cv2.destroyWindow(win_name)
