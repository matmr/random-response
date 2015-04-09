__author__ = 'Matjaz'

import pickle

import numpy as np

import matplotlib.pyplot as plt


fatigue_path = r'\path\to\models'


def frf(W, V1, V2, ind_o, ind_i):
    """Calculate frequency-response-function from the modal model.

    :param W: Diagonal matrix for response calculation.
    :param V1: Eigenvector matrix 1 (response).
    :param V2: Eigenvector matrix 2 (excitation).
    :param ind_o: Indices of dofs on the output.
    :param ind_i: Indices of dofs on the input.
    :return: Frequency-response function relating input to output.
    """
    V1 = V1[ind_o, :]
    V2 = V2[ind_i, :]

    return np.einsum('ij,jkz,lk->ilz', V1, W, V2)


def w_matrix(w, w0, form, damping_type, damping_coefficients=[]):
    """Calculate the diagonal matrix as a part of forced response solution.

    :param w: Frequency values (numpy.ndarray).
    :param w0: Eigenfrequencies (numpy.ndarray).
    :param form: Kinematic meaasure (str: 'a' - acceleration, 'v' - velocity, 'x' - displacement).
    :param damping_type: Damping model (str: 'none', 'vis' - viscous, 'his' - histeretic).
    :param damping_coefficients: Damping coefficients (numpy.ndarray or list).
    """
    nm = w0.size

    if form == 'x':
        wm = np.ones(w.size, dtype='float') * 1.0
    elif form == 'v':
        wm = 1.j * w
    else:
        wm = -w**2

    if damping_type == 'none':
        model = lambda wmx, wx, ind: wmx/(w0[ind]**2 - wx**2)
    elif damping_type == 'his':
        model = lambda wmx, wx, ind: wmx/((1.j*damping_coefficients[ind] + 1.0)*w0[ind]**2 - wx**2)
    elif damping_type == 'vis':
        model = lambda wmx, wx, ind: wmx/(w0[ind]**2 - wx**2 + 1.j*2*w0[ind]*wx*damping_coefficients[ind])
        vis_ewins_model = lambda wmx, wx, ind:  2*wm*(1.j*wx+w0[ind]*damping_coefficients[ind]
                                             )/(w0[ind]**2 - w**2 + 1.j*2*w0[ind]*wx*damping_coefficients[ind])
    else:
        raise ValueError('Unrecognized damping model {0}.'.format(damping_type))

    W = np.zeros((nm, nm, wm.size), dtype='complex128')
    ind = np.arange(0, nm, dtype='int')
    ind_ = np.ix_(ind, ind)

    offset = 0
    if w[0] == 0 and w0[0] == 0:
        W[ind_[0][1:], ind_[1][:, 1:], 0] = np.diag(model(wm[0], w[0], ind[1:]))

        offset = 1

    for i in range(offset, wm.size):
        W[ind_[0], ind_[1], i] = np.diag(model(wm[i], w[i], ind))



    return W

def smurf_base_excitation(w, w0, X, S, response_dofs_idx, excitation_dofs_idx, form='a',
                          damping_type='none', damping_coefficients=[]):
    """Calculate random response to kinematic base-motion excitation.

    :param w: Frequency values (numpy.ndarray).
    :param w0: Eigenfrequencies (numpy.ndarray).
    :param X: Displacement eigenvector matrix.
    :param S: Stress eigenvector matrix.
    :param response_dofs_idx:
    :param excitation_dofs_idx:
    :param form: Kinematic meaasure (str: 'a' - acceleration, 'v' - velocity, 'x' - displacement).
    :param damping_type: Damping model (str: 'none', 'vis' - viscous, 'his' - histeretic).
    :param damping_coefficients: Damping coefficients (numpy.ndarray or list).
    :return: Frequency-response functions.
    """
    W = w_matrix(w, w0, form, damping_type, damping_coefficients)
    Hii = frf(W, X, X, response_dofs_idx, excitation_dofs_idx)
    Hsi = frf(W, S, X, response_dofs_idx, excitation_dofs_idx)

    Hii_inv = np.zeros((Hii.shape[1], Hii.shape[0], Hii.shape[2]), dtype='complex128')
    for i in range(Hii.shape[2]):
        Hii_inv[:, :, i] = np.linalg.pinv(Hii[:, :, i])

    return np.einsum('ijz,jkz->ikz', Hsi, Hii_inv)

def forced_response(w, w0, X, S, response_dofs_idx, excitation_dofs_idx, form='a',
                          damping_type='none', damping_coefficients=[]):
    """Calculate random response to forced excitation.

    :param w: Frequency values (numpy.ndarray).
    :param w0: Eigenfrequencies (numpy.ndarray).
    :param X: Displacement eigenvector matrix.
    :param S: Stress eigenvector matrix.
    :param response_dofs_idx:
    :param excitation_dofs_idx:
    :param form: Kinematic meaasure (str: 'a' - acceleration, 'v' - velocity, 'x' - displacement).
    :param damping_type: Damping model (str: 'none', 'vis' - viscous, 'his' - histeretic).
    :param damping_coefficients: Damping coefficients (numpy.ndarray or list).
    :return:
    """
    W = w_matrix(w, w0, form, damping_type, damping_coefficients)

    return frf(W, S, X, response_dofs_idx, excitation_dofs_idx) # Hsi

if __name__ == '__main__':

    # -- Load data from pickle -- eigenfrequencies, eigenvectors, nodesets ...
    f = open(r'{0}\model_kinematic.pkl'.format(fatigue_path), 'rb')
    data_k = pickle.load(f, encoding='latin1')
    f.close()

    f = open(r'{0}\model_forced.pkl'.format(fatigue_path), 'rb')
    data_f = pickle.load(f, encoding='latin1')
    f.close()

    data_k['w0'] = 2.0 * np.pi * data_k['f0']
    data_f['w0'] = 2.0 * np.pi * data_f['f0']

    kinematic_excitation_node = data_k['nsets']['ASSEMBLY_XNODES'][0]
    force_excitation_nodes = data_k['nsets']['ASSEMBLY_XNODESF']
    response_nodes = data_k['nsets']['ASSEMBLY_RNODES']

    # -- Connect nodeset dofs to the general index (index of all nodes dof).
    response_nodes_dofs_idx = ((response_nodes - 1) * 6).repeat(6) + np.tile(np.arange(6), response_nodes.size)
    response_nodes_dofs_idx = response_nodes_dofs_idx[3000:3010]

    kinematic_excitation_dof_idx = (kinematic_excitation_node-1) * 6 + 1 # node + dof (0-5)
    force_excitation_dof_idx = ((force_excitation_nodes - 1) * 6).repeat(6) + np.tile(np.arange(6), force_excitation_nodes.size)
    force_excitation_dof_idx = force_excitation_dof_idx[2::6]

    # -- Set damping coefficients.
    force_mode_damping = np.ones(data_f['x'].shape[1])*0.003
    force_mode_damping[2] = 0.013
    kinematic_mode_damping = np.ones(data_k['x'].shape[1])*0.003
    kinematic_mode_damping[2] = 0.013

    # -- Calculate FRFs.
    w = np.arange(280*2*np.pi, 580*2*np.pi, 1)

    Ts_base_motion = smurf_base_excitation(w, data_k['w0'], data_k['x'], data_k['s'], response_nodes_dofs_idx,
                                [kinematic_excitation_dof_idx], form='a', damping_type='vis',
                                damping_coefficients=kinematic_mode_damping)

    Ts_forced = forced_response(w, data_f['w0'], data_f['x'], data_f['s'], response_nodes_dofs_idx,
                                force_excitation_dof_idx, form='a', damping_type='vis',
                                damping_coefficients=force_mode_damping)

    # -- Plot.
    for i in range(Ts_forced.shape[0]):
        plt.plot(w/2/np.pi, np.abs(Ts_forced.sum(axis=1)[i, :]/force_excitation_dof_idx.size)+
                 np.abs(Ts_base_motion[i, 0, :]))

    plt.show()
