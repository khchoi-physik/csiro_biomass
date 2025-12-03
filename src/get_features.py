import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def get_rgb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_hsv(path):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def get_gray(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_green_coverage(img_):
    green_mask = (img_[:, :, 1] > img_[:, :, 0]) & (img_[:, :, 1] > img_[:, :, 2])
    return green_mask.sum() / green_mask.size

def get_sobel(gray_, ksize = 3):
    sobelx  = cv2.Sobel(gray_, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely  = cv2.Sobel(gray_, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.sqrt(sobelx**2 + sobely**2)

def get_canny(gray_, low_threshold = 100, high_threshold = 200): 
    return cv2.Canny(gray_, low_threshold, high_threshold) 
    
def get_binary(gray, threshold = 127):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary 
 
def get_lbp(gray_, p = 8, r = 3, method = "uniform"):
    return local_binary_pattern(gray_, P=p, R=r, method=method)

def resized_img(img): 
    return cv2.resize(img, (256, 256))
    
def get_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_glcm(gray_, distance = 1, angle = [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = 256, symmetric = True, normed = True):
    return graycomatrix(gray_, distances=[distance], angles=angle, levels=levels, symmetric=symmetric, normed=normed)

def get_glcm_prop(gray_):
    glcm = get_glcm(gray_)
    
    features = {}

    features['contrast']      = graycoprops(glcm, 'contrast').mean()
    features['dissimilarity'] = graycoprops(glcm, 'dissimilarity').mean()
    features['homogeneity']   = graycoprops(glcm, 'homogeneity').mean()
    features['energy']        = graycoprops(glcm, 'energy').mean()
    features['glcm_corre']    = graycoprops(glcm, 'correlation').mean()
    features['ASM']           = graycoprops(glcm, 'ASM').mean()
    features['entropy']       = graycoprops(glcm, 'entropy').mean()

    return features
