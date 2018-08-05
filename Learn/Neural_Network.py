import numpy as np

# Operation class
class Operation():
    """
    An operation is a node in a Graph
    """
    def __init__(self, input_nodes = []):
        self.intput_nodes = input_nodes
        self.output_nodes = []

        for node in self.input_nodes:
            node.output_nodes.append(self)

        __default_graph.opeartions.append(self)

    def compute(self):
        """
        Placeholder function
        """
        pass

# add Operation
class add(Operation):
    def __init__(self, x, y):
        
        super.__init__(self, [x, y])

    def compute(self, x_var, y_var):

        self.inputs = [x_var, y_var]
        return x_var + y_var

# Muptiply Operation
class multiply(Operation):
    def __init__(self, x, y):

        super.__init__(self, [x, y])

    def compute(self, x_var, y_var):

        self.inputs = [x_var, y_var]
        return x_var * y_var

# Matric Multiplication
class matmul(Operation):
    def __init__(self, x, y):

        super.__init__(self, x, y):

    def compute(self, x_mat, y_mat):

        self.inputs = [x_mat y_mat]
        return x_mat.dot(b_mat)

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


