import cv2
class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        #print(f"I am in {self.x}, {self.y}, I received {del_x}, {del_y}")
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
        #print(f"I just moved to {self.x}, {self.y}")

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
class Wall(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Wall, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("media/brick-wall.png")
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Drone(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Drone, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("media/drone.png")
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Aruco(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Aruco, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("media/aruco.png")
        self.found = 0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))