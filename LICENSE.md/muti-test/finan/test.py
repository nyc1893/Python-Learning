import econci

comp = econci.Complexity(df, c='country', p='product', values='export')
comp.calculate_indexes()
eci = comp.eci
pci = comp.pci

# creating the product space
comp.create_product_space()

# the graphs are networkx.Graph objects
complete_graph = comp.complete_graph  # complete product space
max_spanning_tree = comp.maxst  # maximum spanning tree
prod_space = comp.product_space  # product space

# edges_nodes_to_csv saves one csv file with edges and weights
# and another file with nodes information
econci.edges_nodes_to_csv(prod_space, graph_name='prod_space', dir_path='./data/')