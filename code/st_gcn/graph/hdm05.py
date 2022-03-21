import numpy as np
from . import tools

# Joint index:
# {0,  "HeadTop"}
# {1,  "Head"},
# {2,  "Neck"},
# {3,  "RClavicle"},
# {4,  "RShoulder"},
# {5,  "RElbow"},
# {6,  "RWrist"},
# {7,  "RFingers"},
# {8,  "LClavicle"},
# {9,  "LShoulder"},
# {10, "LElbow"},
# {11, "LWrist"},
# {12, "LFingers"}
# {13, "Chest"}
# {14, "Belly"}
# {15, "Root"}
# {16, "RHip"},
# {17, "RKnee"},
# {18, "RAnkle"},
# {19, "RToes"},
# {20, "LHip"},
# {21, "LKnee"},
# {22, "LAnkle"},
# {23, "LToes"},


# Edge format: (origin, neighbor)
num_node = 24
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (1, 2), (3, 2), (8, 2), (13, 2), (4, 3), (5, 4), (6, 5), (7, 6),  # Top + right_arm
          (9, 8), (10, 9), (11, 10), (12, 11), (14, 13), (15, 14), (16, 15), (20, 15),  # left arm + bottom
          (17, 16), (18, 17), (19, 18), (21, 20), (22, 21), (23, 22)]  # legs
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
    np.set_printoptions(threshold=np.nan)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix())


if __name__ == '__main__':
    main()
