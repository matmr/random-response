__author__ = 'Matjaz'

import pickle

import numpy as np

import matplotlib.pyplot as plt


from modal_two_part.functions import frf, smurf_base_excitation, forced_response



fatigue_path = r'E:\work_matjaz\delovni_synced\matjaz_research\y_sample_multiaxial\fatigue'


if __name__ == '__main__':

    # -- Load data from pickle -- eigenfrequencies, eigenvectors, nodesets ...
    f = open(r'{0}\kinematic_model_hand_2.mod'.format(fatigue_path), 'rb')
    data_k = pickle.load(f, encoding='latin1')
    f.close()

    f = open(r'{0}\kinematic_model_hand_2_f.mod'.format(fatigue_path), 'rb')
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
