class Jammer:
    def __init__(self, jam_radius):
        self.position = (0, 0) # Default
        self.radius = jam_radius
        self.active = 0 # Default 0: inactive, 1: active
        self.is_destroyed = False

    def set_position(self, x, y):
        self.position = (x, y)
    
    def current_position(self):
        return self.position

    def set_destroyed(self):
        self.is_destroyed = True
    
    def get_destroyed(self):
        return self.is_destroyed