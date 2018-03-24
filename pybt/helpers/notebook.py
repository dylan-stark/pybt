"""Some helpful methods for working with notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt

def member_observations(obs, index):
    """Compile member observations into a table.
    """

    step, epoch, acc, loss, val_acc, val_loss = ([], [], [], [], [], [])

    member = obs[index]

    for x in member['observations']:
        step.extend([x['t'] for i in range(len(x['epochs']))])
        epoch.extend(x['epochs'])
        acc.extend(x['acc'])
        loss.extend(x['loss'])
        val_acc.extend(x['val_acc'])
        val_loss.extend(x['val_loss'])

    return step, epoch, acc, loss, val_acc, val_loss

def learning_rates(obs):
    """Compile learning rate settings by epoch into a table.
    """

    epoch, lr = ([], [])

    for m in obs:
        for o in m['observations']:
            epoch.extend(o['epochs'])
            lr.extend(o['learning_rate'])

    return epoch, lr

def batch_sizes(obs):
    """Compile batch size settings by epoch into a table.
    """

    epoch, batch_size = ([], [])

    for m in obs:
        for o in m['observations']:
            epoch.extend(o['epochs'])
            batch_size.extend(o['batch_size'])

    return epoch, batch_size

def plot_all_acc(obs, nrow=None, ncol=3):
    """Plot training and validation accuracy for all population members.
    """

    def plot_member_panel_acc(panel, obs, index):
        step, epoch, acc, loss, val_acc, val_loss = \
            member_observations(obs, index)

        panel.text(18, .9, '{}'.format(index), fontsize=24)
        panel.plot(epoch, acc, 'bo', label='Training acc.')
        panel.plot(epoch, val_acc, 'b', label='Validation acc.')

    num_members = len(obs)

    if nrow == None and ncol == None:
        raise ValueError('at least one of `nrow` and `ncol` must be set')

    if nrow == None:
        nrow = num_members // ncol + (num_members % ncol != 0)
    elif ncol == None:
        ncol = num_members // nrow + (num_members % nrow != 0)

    if num_members > nrow * ncol:
        raise RuntimeError('cannot fit {} plots in {} panels'.format(
            num_members, nrow*ncol))

    plt.close('all')

    fig, panels = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    m_id = 0
    for i in range(nrow):
        for j in range(ncol):
            if m_id < num_members:
                panel = panels[i,j]
                plot_member_panel_acc(panel, obs, m_id)
                m_id += 1

    plt.tight_layout()

def plot_members_acc(obs):
    """Plot training and validation accuracy for all members.
    """

    for i in range(len(obs)):
        step, epoch, acc, loss, val_acc, val_loss = member_observations(obs, i)

        plt.plot(epoch, acc, 'bo', label='Training acc.')
        plt.plot(epoch, val_acc, 'b', label='Validation acc.')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training acc.', 'Validation acc.'])

    return plt

def plot_member_acc(obs, index):
    """Plot training and validation accuracy for specified member.
    """

    step, epoch, acc, loss, val_acc, val_loss = member_observations(obs, index)

    plt.plot(epoch, acc, 'bo', label='Training acc.')
    plt.plot(epoch, val_acc, 'b', label='Validation acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    return plt

def plot_member_loss(obs, index):
    """Plot training and validation loss for specified member.
    """

    step, epoch, acc, loss, val_acc, val_loss = member_observations(obs, index)

    plt.plot(epoch, loss, 'bo', label='Training loss')
    plt.plot(epoch, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    return plt

def plot_learning_rate(obs):
    """Plot log of learning rates.
    """

    epoch, lr = learning_rates(obs)
    lr = [np.log10(x) for x in lr]

    plt.scatter(epoch, lr)
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate (log10)')

    return plt

def plot_batch_size(obs):
    """Plot batch sizes.
    """

    epoch, bs = batch_sizes(obs)

    plt.scatter(epoch, bs)
    plt.xlabel('Epochs')
    plt.ylabel('Batch size')

    return plt
