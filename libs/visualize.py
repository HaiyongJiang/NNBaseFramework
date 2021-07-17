import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def cuboid_data(o, size=(1,1,1), theta=0):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float) # [6, 4, 3]
    # scale
    S = np.array(size).astype(float) # [3,]
    X *= S
    X -= S/2 # [6,4,3]
    # rotate
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1],
                  ], dtype=np.float) # [3,3]
    X = np.matmul(X.reshape((6*4,3)), R.transpose(1,0)).reshape(6,4,3) # [6, 4, 3]
    X += S/2
    X += np.array(o)
    return X

def plotCubeAt(positions, sizes, thetas=None, c='r', alpha=1.0, **kwargs):
    if thetas is None:
        thetas = np.zeros((positions.shape[0], 1), dtype=np.float)
    g = []
    for p,s,t in zip(positions,sizes,thetas):
        g.append( cuboid_data(p, size=s, theta=t) )
    return Poly3DCollection(np.concatenate(g), facecolors=c, alpha=alpha, **kwargs)

def visualize_sparse_voxels(voxels, corners, radius, out_file=None):
    r''' Visualizes voxel data.

    Args:
        voxels (np.ndarray): voxel data, [N,3]
        corners (np.ndarray): corner for each voxel, [N, 8, 3]
        out_file (string): output file
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    N = voxels.shape[0]
    sizes = np.ones_like(voxels)*radius
    # cubes = plotCubeAt(voxels, sizes, c=[0.0, 1.0, 0.0, 0.4])
    # ax.add_collection3d(cubes)

    ax.scatter(voxels[:, 2:3], voxels[:, 0:1], voxels[:, 1:2], c='g', marker='o', s=1.0)
    corners = corners.reshape((-1, 3))
    ax.scatter(corners[:, 2:3], corners[:, 0:1], corners[:, 1:2], c='r', marker='o', s=4.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, color="y"):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=2.0, c=color)
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def visualize_pointcloud_layer(points, preds, out_file=None):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ratios = [[0.0, 0.2], [0.2,0.4], [0.4,0.5], [0.5, 0.6]]
    clrs = ["g", "c", "r", "b"]
    preds = torch.sigmoid(preds)
    for ii,r in enumerate(ratios):
        mask = ((preds>=1.0-r[1]) & (preds<1.0-r[0]))
        points_sel = points[torch.nonzero(mask).reshape(-1), :].data.cpu().numpy()
        ax.scatter(points_sel[:, 2], points_sel[:, 0], points_sel[:, 1], s=2.0, c=clrs[ii])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)

