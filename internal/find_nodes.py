import numpy as np
import matplotlib.pyplot as plt

### Calculates measurement node positions for an input model using a custom marching spheres algorithm

### Part of the VTSim Package

def sphere_fill(boundaries, source_node, empty_space, node_separation, sim_output_path):
    boundaries = boundaries[:,:-empty_space-1,:]
    empty_space_coords = np.stack(np.where(boundaries==0),axis=1)

    nodes = [source_node]

    ### find all empty space coords that are a set distance from the source node
    i = 0
    while True:
        shell_points_ids = np.where(np.abs(np.linalg.norm(empty_space_coords-nodes[i],axis=1)-node_separation)<=0.5)
        shell_points = empty_space_coords[shell_points_ids]
        if i > 0:
            for j in range(len(nodes)-1):
                shell_points_ids = np.where(np.linalg.norm(shell_points-nodes[j],axis=1)>=node_separation*1.3)
                shell_points = shell_points[shell_points_ids]
        if shell_points.size == 0:
            break
        average_shell_loc = np.round(np.average(shell_points, axis=0)).astype(int)
        while not (np.any(np.all(shell_points == average_shell_loc, axis=1))) and (shell_points.size >= 0):
            # closest_shell_points_ids = np.argsort(np.linalg.norm(shell_points-average_shell_loc, axis=1))[0]
            # average_shell_loc = shell_points[closest_shell_points_ids]

            closest_shell_points_ids = np.argsort(np.linalg.norm(shell_points-average_shell_loc, axis=1))[:-1]
            shell_points = shell_points[closest_shell_points_ids]
            average_shell_loc = np.round(np.average(shell_points, axis=0)).astype(int)
        if shell_points.size == 0:
            print('Node never found.')
            quit()
        elif np.any(np.all(shell_points == average_shell_loc, axis=1)):
            nodes.append(tuple(average_shell_loc))
        else:
            print('uhoh')
            quit()
        i+=1

    print('Nodes found:', len(nodes))

    nodes = np.array(nodes)
    propagation_nodes_ids = np.where(nodes[:,1]>=boundaries.shape[1]*0.9)
    print('Redundant propagation space nodes removed:',len(propagation_nodes_ids))
    nodes = np.delete(nodes, propagation_nodes_ids[0], axis=0)
    propagation_node = [np.round((boundaries.shape[0]-1)/2), boundaries.shape[1]-1, nodes[-1][2]]
    nodes = np.vstack((nodes, propagation_node)).astype(int)

    visual_node_confirmation(sim_output_path, nodes, boundaries, empty_space)

    np.savetxt(sim_output_path+'\\nodes.txt',nodes, fmt='%i', delimiter=',')
    nodes = [tuple(i) for i in nodes]

    print('Visual confirmation complete.')

    return nodes

def visual_node_confirmation(sim_output_path, nodes, boundaries, empty_space):
    boundaries = boundaries[:,:-empty_space-1,:]
    empty_space_coords = np.stack(np.where(boundaries==0),axis=1)

    print('Number of nodes for sim:',len(nodes))
    print('Visual confirmation (nodes in red)')

    ### Visual Check
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')
    empty = ax3D.scatter(empty_space_coords[:,0],empty_space_coords[:,1],empty_space_coords[:,2],s=2, alpha=0.025, color='0.4')
    nodes = ax3D.scatter(nodes[:,0],nodes[:,1],nodes[:,2], s=70 , color = '0', alpha=1)
    ax3D.view_init(elev=20, azim=45)
    plt.savefig(sim_output_path+"\\NodeLocationVisualisation.pdf",bbox_inches='tight', dpi=300)
    empty.set_color('C0')
    nodes.set_color('red')
    plt.savefig(sim_output_path+"\\NodeLocationVisualisation.png",bbox_inches='tight', dpi=300)
    plt.show()
