from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def plot_contour(fig_number, name, vp: np.array, vp_range=None, fill=False, extent=None,
    cmap='jet_r', levels=None, colors=None, save=None, ** plot_kwargs):

    plt.figure(fig_number, dpi=300)

    if vp_range is None:
        vp_range = (None, None)

    if colors is not None:
        cmap = None

    if levels is None:
        levels = np.linspace(np.min(vp), np.max(vp), 5)

    if extent is None:
        extent = (1000.0, 1000.0)

    extent = np.array(extent)

    if fill:
        func = plt.contourf
    else:
        func = plt.contour

    plot = func(
        vp.T,
        cmap=cmap,
        extent=(0.0, extent[0], 0.0, extent[1]),
        vmin=vp_range[0], vmax=vp_range[1],
        colors=colors,
        levels=levels,
        **plot_kwargs
    )

    plt.xlim(0.0, extent[0])
    plt.ylim(extent[1], 0.0)
    plt.xlabel('X (km)')
    plt.ylabel('Z (km)')

    plt.gca().set_aspect('equal')
    if save:
        plt.savefig(f'../../FWI_Python/plots/{name}.png')
    plt.close()

    return


def plot_inDomain(target,mesh,field_name,ID):

    nNodesL = mesh.nElementsL + 1
    nNodesD = mesh.nElementsD + 1

    aux_fd = np.zeros((mesh.nNodes, 1))
    for i in range(0, target.shape[0]):
        aux_fd[target[i] - 1, 0] = 1

    axField = np.zeros([nNodesD, nNodesL])

    for j in range(0, nNodesD):
        for i in range(0, nNodesL):
            axField[(nNodesD - 1) - j, i] = aux_fd[i + j * nNodesL, 0]

    fig1, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(axField, cmap = 'binary')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.title(' ')
    plt.savefig(f'../../FWI_Python/plots/{field_name}_{ID}.png')
    plt.close(fig1)

def plot_field(mesh,field,ncol,field_name,ID):

    # Helpful info: mesh  = meshObj
    #               field = p x 1 (ncol = 0) or p x t(specific ncol)
    #               field_name = string to name the saved im
    #               ID = another code to name the saved im

    aux      = np.zeros([mesh.nElements ,1])
    axField = np.zeros([mesh.nElementsD, mesh.nElementsL])

    for e in range(0 ,mesh.nElements):
        counter = 0
        for node in range(0, 4):
            counter += 0.25*field[mesh.Connect[e, node] - 1, ncol]
        aux[e ,0] = counter

    for j in range(0, mesh.nElementsD):
        for i in range(0, mesh.nElementsL):
            axField[(mesh.nElementsD - 1) - j, i] = aux[i + j * mesh.nElementsL, 0]

    fig1, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(axField, cmap = 'binary')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.title(' ')
    plt.savefig(f'../../FWI_Python/plots/{field_name}_{ID}.png')
    plt.close(fig1)


def plot_cost(cF,it,niter):

    x = np.linspace(1, len(cF), len(cF), dtype=int)
    y = np.zeros([len(cF), 1])
    for i in range(0, len(cF)):
        y[i,0] = cF[i]
    fig1 = plt.figure(figsize=(7, 7))
    ax = fig1.gca()
    plt.plot(x, y, color='b', marker='o', label=' ')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.draw()
    plt.grid(True)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.title(' ')
    if it >= niter - 2:
        plt.savefig('../../FWI_Python/plots/costFunctional.png')
    plt.show(block=False)
    plt.pause(.3)

    return fig1


def render_propagating(mesh,field,size):

    #Helpful info: mesh  = meshObj
    #              field = p x t
    #              size  = size of dt analysed

    ims = []
    steps = field.shape[1]//size

    aux = np.zeros([mesh.nElements, 1])
    axField = np.zeros([mesh.nElementsD, mesh.nElementsL])

    fig1, ax = plt.subplots(figsize=(7, 7))

    for t in range(0, steps):
        time = size * t
        if time < field.shape[1]:
            for e in range(0, mesh.nElements):
                counter = 0
                for node in range(0, 4):
                    counter += 0.25 * field[mesh.Connect[e, node] - 1, time]
                aux[e, 0] = counter

            for j in range(0, mesh.nElementsD):
                for i in range(0, mesh.nElementsL):
                    axField[(mesh.nElementsD - 1) - j, i] = aux[i + j * mesh.nElementsL, 0]

            im = ax.imshow(axField, cmap='binary', animated=True)
            ims.append([im])

    ani = animation.ArtistAnimation(fig1,ims,interval=5,blit=True,repeat_delay=800)

    plt.show()


def plot_control(mesh,model,lsmin,lsmax):

    nx = mesh.nElementsL+1
    ny = mesh.nElementsD+1

    axField = np.zeros([nx,ny])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(1, mesh.nElementsL+2, 1)
    Y = np.arange(1, mesh.nElementsD+2, 1)
    X, Y = np.meshgrid(X, Y)

    for j in range(0, ny):
        for i in range(0, nx):
            axField[ny - 1 - j, i] = model[i + j * ny, 0]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, axField, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(lsmin, lsmax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

