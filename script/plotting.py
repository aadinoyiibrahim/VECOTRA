"""
VECOTRA : VElocity Constrained Optinmal TRAnsport
    Constributors:
                Abdullahi Ibrahim
                Caterina De Bacco
"""

'''
plotting the network
'''
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
    

def plot_graphs(graph,flux_norm1,w1 = 0.01, 
            figsize=(20,7),ns = 10,outfigure=None,
            colors_map = {0:'b',1:'r'},
            algo_labels = {'MultiOT_UConstr':'UnConstrained'}, 
            fs_title = 20, dpi = 300):
    
    algos = list(flux_norm1.keys())
    fig, axes = plt.subplots(nrows=1, ncols=len(algos),figsize=figsize)
    ax = axes.flatten()
    pos_plot = nx.get_node_attributes(graph,'pos')
    
    for a_id,a in enumerate(algos):
        widths = [w1 * flux_norm1[a][idx] for idx, e in enumerate(list(graph.edges())) ]
        nx.draw(graph,pos_plot, node_size = ns, with_labels = False, edge_color='blue', font_weight="bold",width = widths, ax=ax[a_id])
        ax[a_id].set_title(algo_labels[a],fontsize = fs_title)

    if outfigure is not None:
        plt.savefig(outfigure + '.png', dpi=dpi, format='png', bbox_inches='tight',pad_inches=0.1)
    return plt.show(),plt.gcf()

def plot_g(tdens1, tdens2, capacity, figsize=(14, 6), outfig=None):
    fs = 30
    y1 = np.array(tdens1)
    y2 = np.array(tdens2)
    x = np.arange(len(y1))
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(1,2,figsize=figsize)
    ax[0].plot(x, y1)
    ax[0].plot(x, capacity * np.ones(len(x)), ls="--", c="grey")
    ax[1].plot(x, y2)
    ax[1].plot(x, capacity * np.ones(len(x)), ls="--", c="grey")
    ax[0].set_title("Unconstrained",fontsize=fs)
    ax[1].set_title("Constrained",fontsize=fs)
    
    for i in list(range(0,2)):
        ax[i].set_xlabel("Edges",  fontsize=fs)
        ax[0].set_ylabel(r"$ \mu_e$", fontsize=fs)
        ax[i].text(-17, capacity[0], r'$c_e$', {'color': 'grey', 'fontsize': 25})
    plt.tight_layout()

    return plt.show()

def plot_cost(cost_uconstr, cost_constr, cost_clip):
    
    
    figsize = (20,5)
    ms = 10
    fs = 20
    fig, ax = plt.subplots(1,3,figsize=figsize)
    ax[0].plot(np.arange(cost_uconstr.size)[:100], cost_uconstr[:100], alpha=.4, marker="o", color='green', ms=ms, label="unconstrained")
    ax[1].plot(np.arange(cost_constr.size)[:100], cost_constr[:100], alpha=.4, marker="s", ms=ms, label="capacity")
    ax[2].plot(np.arange(cost_clip.size)[:100], cost_clip[:100], alpha=.4, marker="^",color='grey', ms=ms, label="clip")
    
    
    for i in list(range(0,3)):
        ax[i].legend(fontsize=20, labelspacing=0)
        ax[i].set_xlabel("Iteration", fontsize=fs)
        ax[0].set_ylabel(r"Cost", fontsize=fs)
    plt.show()
