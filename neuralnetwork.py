import numpy as np

class NeuralNet(object):
    def __init__self():
        self.input_layer_size = 1
        self.hidden_size=2
        self.output_size=2
        
    def relu(z):
        return max(0,z)

    def init_weights():
        self.Wh = np.random.randn(input_layer_size,hidden_layer_size) * \
             np.sqrt(2.0/input_layer_size)
        self.Wo = np.random.randn(hidden_later_size)
        
    #wh weight hidden wo weight output
    def feed_forward(x,Wh,Wo):
        #hidden
        Zh = x * Wh
        H = relu(Zh)
        #output
        Zo = H * Wo
        output = relu(Zo)
        return output

