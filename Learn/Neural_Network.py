import numpy as np

# Operation class
class Operation():
    """
    An operation is a node in a Graph
    """
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []
    
    # Not Fully Understand here
        for node in self.input_nodes:
            node.output_nodes.append(self)
        _default_graph.operations.append(self)

    def compute(self):
        """
        Placeholder function
        """
        pass

# add Operation
class add(Operation):
    def __init__(self, x, y):
        
        super().__init__([x, y])

    def compute(self, x_var, y_var):

        self.inputs = [x_var, y_var]
        return x_var + y_var

# Muptiply Operation
class multiply(Operation):
    def __init__(self, x, y):

        super().__init__([x, y])

    def compute(self, x_var, y_var):

        self.inputs = [x_var, y_var]
        return x_var * y_var

# Matric Multiplication
class matmul(Operation):
    def __init__(self, x, y):

        super().__init__([x, y])

    def compute(self, x_mat, y_mat):

        self.inputs = [x_mat, y_mat]
        return x_mat.dot(b_mat)

# Get the Sigmoid Operation
class Sigmoid(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_var):
        return 1 / (1 + np.exp(-x_var))


### Placeholders
class Placeholder():
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """

    def __init__(self):
        
        self.output_nodes = []

        _default_graph.placeholders.append(self)

### Variables
class Variable():
    """
    This variable is a changeable parameter of the Graph.
    """
    
    def __init__(self, initial_value = None):
        
        self.value = initial_value

        self.output_nodes = []

        _default_graph.variables.append(self)

## Graph
class Graph():
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """
        PostOrdr Traverse of Nodes. Make computation are done correstly
    """
    node_postorder = []
    recurse(operation, node_postorder)
    return node_postorder

def recurse(node, node_postorder):
    if isinstance(node, Operation):
        for input_node in node.input_nodes:
            recurse(input_node, node_postorder)
    node_postorder.append(node)

### Run the Session
class Session():
    def run(self, operation, feed_dict={}):

        node_postorder = traverse_postorder(operation)

        for node in node_postorder:
            if type(node) == Variable:
                node.output = node.value
            elif type(node) == Placeholder:
                node.output = feed_dict[node]
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output

if __name__ == '__main__':
    g = Graph()
    g.set_as_default()

    A = Variable(10)
    b = Variable(1)
    x = Placeholder()
    y = multiply(A, x)
    z = add(y, b)

    sess = Session()
    result = sess.run(z, {x:[[10, 20],[20, 30]]})
    print(result)






