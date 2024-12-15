import numpy as np


def compute_system_matrix(b_value, b_vectors, order):
    magnitudes = np.linalg.norm(b_vectors, axis=1)
    unit_vectors = b_vectors / magnitudes[:, np.newaxis]
    b_vectors = unit_vectors
    B = np.zeros((b_vectors.shape[0], 6))

    if order == 'fsl':
        B[:, 0] = b_vectors[:, 0]*b_vectors[:, 0]*b_value #xx
        B[:, 1] = 2*b_vectors[:, 0]*b_vectors[:, 1]*b_value #xy
        B[:, 2] = 2*b_vectors[:, 0]*b_vectors[:, 2]*b_value #xz
        B[:, 3] = b_vectors[:, 1]*b_vectors[:, 1]*b_value #yy
        B[:, 4] = 2*b_vectors[:, 1]*b_vectors[:, 2]*b_value #yz
        B[:, 5] = b_vectors[:, 2]*b_vectors[:, 2]*b_value #zz
    
    elif order == 'mrtrix':
        B[:, 0] = b_vectors[:, 0]*b_vectors[:, 0]*b_value #xx
        B[:, 1] = b_vectors[:, 1]*b_vectors[:, 1]*b_value #yy
        B[:, 2] = b_vectors[:, 2]*b_vectors[:, 2]*b_value #zz
        B[:, 3] = 2*b_vectors[:, 0]*b_vectors[:, 1]*b_value #xy
        B[:, 4] =  2*b_vectors[:, 0]*b_vectors[:, 2]*b_value #xz
        B[:, 5] = 2*b_vectors[:, 1]*b_vectors[:, 2]*b_value #yz

    return B


def fit_dti(dwi_data, B):
    dwi_flatten = dwi_data.reshape(-1, dwi_data.shape[3])
    dwi_flatten = np.maximum(dwi_flatten, np.min(dwi_flatten[dwi_flatten>0]))

    S0 = dwi_flatten[:, 0]
    S = dwi_flatten[:, 1:]
    B_dot = np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T)
    X = - np.log((S)/(S0[:, np.newaxis]))

    D = np.dot(B_dot, X.T)
    D = D.T
    dti = D.reshape(dwi_data.shape[0], dwi_data.shape[1], dwi_data.shape[2], 6)
    dti = np.nan_to_num(dti, nan=0)
    dti[dti==np.max(dti)] = 0
    dti[dti==np.min(dti)] = 0
    dti = dti.astype(np.float32)
    return dti


def construct_D_matrix(dti, i, j, k, order):
    if order ==  'fsl' :
        Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = dti[i, j, k]
    elif order == 'mrtrix' :
        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = dti[i, j, k]

    D_voxel = np.array([
        [Dxx, Dxy, Dxz],
        [Dxy, Dyy, Dyz],
        [Dxz, Dyz, Dzz]
    ])
    return D_voxel


def SVD_dti(dti, order) : 
    evals = np.zeros((dti.shape[0], dti.shape[1], dti.shape[2], 3), dtype = np.float32)  # Shape (128, 128, 60, 3)
    evecs = np.zeros((dti.shape[0], dti.shape[1], dti.shape[2], 3, 3), dtype = np.float32)  # Shape (128, 128, 60, 3, 3)
    peaks = np.zeros((dti.shape[0], dti.shape[1], dti.shape[2], 3, ), dtype = np.float32)  # Shape (128, 128, 60, 3)

    for i in range(dti.shape[0]):
        for j in range(dti.shape[1]):
            for k in range(dti.shape[2]):

                D_voxel = construct_D_matrix(dti, i, j, k, order)

                vals, vecs = np.linalg.eigh(D_voxel)

                evals[i, j, k, :] = vals
                evecs[i, j, k, :, :] = np.array([vec / np.linalg.norm(vec) for vec in vecs])
                peaks[i, j, k, :] = vecs[:, np.argmax(vals)]

    return evals, evecs, peaks


def compute_dti_metrics(dti, eigenvalues, peaks, order):
    shape = eigenvalues.shape[:3]
    fa = np.zeros(shape, dtype=np.float32)
    md = np.zeros(shape, dtype=np.float32)
    ad = np.zeros(shape, dtype=np.float32)
    rd = np.zeros(shape, dtype=np.float32)
    adc = np.zeros(shape, dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                vals = eigenvalues[i, j, k]

                if np.all(vals == 0):
                    fa[i, j, k] = 0  # FA should remain 0
                else:
                    fa[i, j, k] = np.sqrt(0.5) * np.sqrt(
                        ((vals[0] - vals[1]) ** 2 + 
                         (vals[0] - vals[2]) ** 2 + 
                         (vals[1] - vals[2]) ** 2) / 
                        (vals[0] ** 2 + vals[1] ** 2 + vals[2] ** 2)
                    )
                
                # Mean diffusivity
                md[i, j, k] = np.mean(vals)
                
                # Axial diffusivity
                ad[i, j, k] = np.max(vals)
                
                # Radial diffusivity
                rd[i, j, k] = np.mean(np.partition(vals, 1)[:2])
                
                # Apparent diffusion coefficient
                D = dti[i, j, k]
                if order == 'mrtrix':
                    adc[i, j, k] = (D[0] + D[1] + D[2]) / 3
                elif order == 'fsl':
                    adc[i, j, k] = (D[0] + D[3] + D[5]) / 3

    # Replace negative values with 0
    fa = np.maximum(fa, 0)
    md = np.maximum(md, 0)
    ad = np.maximum(ad, 0)
    rd = np.maximum(rd, 0)
    adc = np.maximum(adc, 0)

    # RGB image scaled by FA and peaks
    rgb = fa[..., np.newaxis] * peaks

    return fa, md, ad, rd, rgb, adc