


class Source:

    def __init__(self,id,x,y,mesh,active=True):

        self.id = id

        #Spacial position
        self.x = x
        self.y = y
        self.active = active

        #Mesh position
        side_ratio = self.x / mesh.length
        if side_ratio > 1.0:
            side_ratio = 1.0
        if side_ratio < 0.0:
            side_ratio = 0.0
        self.nodalX = 1 + round(side_ratio * mesh.nElementsL)
        if self.nodalX == 0:
            self.nodalX = 1

        side_ratio = self.y / mesh.depth
        if side_ratio > 1.0:
            side_ratio = 1.0
        if side_ratio < 0.0:
            side_ratio = 0.0
        self.nodalY = 1 + round(side_ratio * mesh.nElementsD)
        if self.nodalY == 0:
            self.nodalY = 1

        self.nodalAbs = self.nodalX + (mesh.nElementsL + 1) * (self.nodalY - 1)


    def on_off(self,turn):

        self.active = turn


    def print_source(self):

        print("------------------------------------")
        print(f"Source {self.id}")
        print(f"Active: {self.active}")
        print(f"x = {self.x}")
        print(f"y = {self.y}")
        print(f"nodalX = {self.nodalX}")
        print(f"nodalY = {self.nodalY}")
        print(f"nodalAbs = {self.nodalAbs}")
        print("------------------------------------")