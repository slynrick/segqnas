import importlib


class Layer(object):
    def __init__(self, cell, block, kernel_size, filters):
        self.cell = self._instantiate_cell(cell, block, kernel_size, filters)

    def _instantiate_cell(self, cell, block, kernel_size, filters):
        module = importlib.import_module("cnn.cells")
        class_ = getattr(module, cell)
        return class_(block, kernel_size, filters)

    def __call__(self, inputs, name=None, is_train=True):
        return self.cell(inputs, name=name, is_train=is_train)