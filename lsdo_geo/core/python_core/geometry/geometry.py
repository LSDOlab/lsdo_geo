

import m3l

class Geometry(m3l.Function):
    
    # def __init__(self, function_space, coefficients):
    #     self.function_space = function_space
    #     self.coefficients = coefficients

    def get_function_space(self):
        return self.function_space
    
    def define_component(self):
        pass

    def refit(self):
        pass

    def plot(self):
        pass




if __name__ == "__main__":
    geometry = Geometry()