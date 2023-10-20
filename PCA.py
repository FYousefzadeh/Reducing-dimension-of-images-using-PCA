import numpy as np
import pandas as pd
import cv2
from skimage import color
import matplotlib.pyplot as plt
from glob import iglob
from sklearn.metrics import mean_squared_error


def load_images():
    faces = pd.DataFrame([])
    for path in iglob(r'jaffe\*.tiff'):
        img = cv2.imread(path)
        img_gray = color.rgb2gray(img)
        img_resize = cv2.resize(img_gray, dsize=(64, 64))
        face = pd.Series(img_resize.flatten(), name=path)
        faces = faces.append(face)
    return faces


# visualize data
def visualize_data(faces):
    faces = np.array(faces)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(np.reshape(faces[0, :], (64, 64)), cmap='gray')
    axarr[0, 1].imshow(np.reshape(faces[1, :], (64, 64)), cmap='gray')
    axarr[1, 0].imshow(np.reshape(faces[2, :], (64, 64)), cmap='gray')
    axarr[1, 1].imshow(np.reshape(faces[3, :], (64, 64)), cmap='gray')
    f.show()
    return


# pre-processing
def normalize_data(data):
    sigma = np.std(data, axis=0)
    mu = np.mean(data, axis=0)
    normalized_data = np.divide(data - mu, sigma)
    return normalized_data, sigma, mu


# PCA
def pca(normalized_faces, sigma, mu, k, v):
    normalized_faces = np.array(normalized_faces)
    cov_mat = np.cov(normalized_faces.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    ind_sort = np.argsort(-abs(eig_vals))  # sort eig_vals in descending order
    ind_eig_vects = ind_sort[0:k]
    w = eig_vecs[:, ind_eig_vects]
    # reconstruct data
    reconst_faces = normalized_faces @ w @ w.T
    sigma_rep = np.transpose([sigma] * 170)
    mu_rep = np.transpose([mu] * 170)
    reconst_faces1 = reconst_faces * sigma_rep.T + mu_rep.T
    faces = normalized_faces * sigma_rep.T + mu_rep.T
    if v == 1:
        reconst_imgs = np.reshape(reconst_faces1, (-1, 64, 64))
        # visualize reconstructed images
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(abs(reconst_imgs[0, :, :]), cmap='gray')
        axarr[0, 1].imshow(abs(reconst_imgs[1, :, :]), cmap='gray')
        axarr[1, 0].imshow(abs(reconst_imgs[2, :, :]), cmap='gray')
        axarr[1, 1].imshow(abs(reconst_imgs[3, :, :]), cmap='gray')
        f.show()
    return reconst_faces1, faces


def find_best_k(normalized_faces):
    normalized_faces = np.array(normalized_faces)
    cov_mat = np.cov(normalized_faces.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    ind_sort = np.argsort(-abs(eig_vals))  # sort eig_vals in descending order
    mse = np.zeros(500)
    # visualize 2D and  3D dataset
    ind_eig_vects = ind_sort[0:3]
    w = eig_vecs[:, ind_eig_vects]
    # dimension reduction
    dim_reduced = normalized_faces @ w
    sigma_rep = np.transpose([sigma[ind_eig_vects]] * 170)
    mu_rep = np.transpose([mu[ind_eig_vects]] * 170)
    dim_reduced1 = dim_reduced * sigma_rep.T + mu_rep.T
    # 2D plot
    f1 = plt.figure()
    plt.scatter(dim_reduced[:, 0], dim_reduced[:, 1])
    f1.show()
    # 3D plot
    f2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(dim_reduced[:, 0], dim_reduced[:, 1], dim_reduced[:, 2])
    f2.show()
    for k in range(500):
        ind_eig_vects = ind_sort[0:k]
        w = eig_vecs[:, ind_eig_vects]
        # reconstruct data
        reconst_faces = normalized_faces @ w @ w.T
        mse[k] = mean_squared_error(abs(reconst_faces), abs(normalized_faces))
        if k == 1:
            reconst_imgs = np.reshape(reconst_faces, (-1, 64, 64))
            # visualize first eigen faces
            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].imshow(abs(reconst_imgs[0, :, :]), cmap='gray')
            axarr[0, 1].imshow(abs(reconst_imgs[1, :, :]), cmap='gray')
            axarr[1, 0].imshow(abs(reconst_imgs[2, :, :]), cmap='gray')
            axarr[1, 1].imshow(abs(reconst_imgs[3, :, :]), cmap='gray')
            f.show()
    diff = np.diff(mse)
    k_best = np.where(abs(diff) < 10e-5)[0][0]
    f = plt.figure()
    plt.plot(mse)
    f.show()
    return k_best

faces = load_images()
normalized_faces, sigma, mu = normalize_data(faces)
k = find_best_k(normalized_faces)
visualize_data(faces)
pca(normalized_faces, sigma, mu, 1, 1)
pca(normalized_faces, sigma, mu, 40, 1)
pca(normalized_faces, sigma, mu, 120, 1)
print("end")
