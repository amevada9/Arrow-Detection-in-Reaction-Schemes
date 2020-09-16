# arrow.py
class Arrow():
    def __init__(self, contour, centroid, average, direction):
        self.contour = contour
        self.centroid = centroid
        self.average = average
        self.direction = direction
        
    def to_dictionary(self):
        info = {}
        info['Centroid'] = self.centroid
        info['Average'] = self.average
        info['Direction'] = self.direction
        return info