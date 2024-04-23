class Jammer:
    def __init__(self, jam_radius):
        self.position = (0, 0) # Default
        self.radius = jam_radius
        self.active = 0 # Default 0: inactive, 1: active

    def set_position(self, x, y):
        self.position = (x, y)
    
    def current_position(self):
        return self.position