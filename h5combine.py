import h5py


class combine(object):
    list_net_a_layer = []
    list_net_b_layer = []

    def __init__(self, net_a, net_b, new_net):
        self.net_a = net_a
        self.net_b = net_b
        self.new_net = new_net

    def append_in_net_a(self, name):
        self.list_net_a_layer.append(name)

    def append_in_net_b(self, name):
        self.list_net_b_layer.append(name)

    def print_name(name):
        print(name)

    def combine(self):
        new_net_path = self.new_net.get_folder_name() + 'weights.hdf5'
        net_a_path = self.net_a.get_folder_name() + 'weights.hdf5'
        net_b_path = self.net_b.get_folder_name() + 'weights.hdf5'
        new_net_file = h5py.File(new_net_path, 'w')
        net_a_file = h5py.File(net_a_path, 'r')
        net_b_file = h5py.File(net_b_path, 'r')
        net_a_file.visit(self.append_in_net_a)
        net_b_file.visit(self.append_in_net_b)
        list_add_layers_name = []
        for layer in self.new_net.list_layers:
            if layer.name not in list_add_layers_name:
                if layer in self.net_a.list_layers:
                    net_a_file.copy('model_weights/' + layer.name, new_net_file)
                    list_add_layers_name.append(layer.name)
                else:
                    net_b_file.copy('model_weights/' + layer.name, new_net_file)
                    list_add_layers_name.append(layer.name)
