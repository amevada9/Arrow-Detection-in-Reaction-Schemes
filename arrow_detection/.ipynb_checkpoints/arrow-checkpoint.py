'''
Basic class with method for determining what an arrow is
Defined by its contour, centroid, average and direction

WE can use this to store information about the arrow in the JSON
output and keep it as a dictionary
'''
class Arrow():
    def __init__(self, contour, centroid, average, direction):
        self.contour = contour
        self.centroid = centroid
        self.average = average
        self.direction = direction
        
    def to_dictionary(self):
        '''
        Converts an Arrow Object into a dictionary that can 
        be easy to work with and place
        Used in JSON insertion to allow each arrow to have
        its own organized catagories. 
        
        @PARAM:
            - self: acts on an arrow object
        @RETURN:
            - info: dictionary containing the centroid, average and direction 
                    of the arrow
        '''
        info = {}
        info['Centroid'] = self.centroid
        info['Average'] = self.average
        info['Direction'] = self.direction
        return info