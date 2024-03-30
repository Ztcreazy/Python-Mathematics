import math

"""
latitude: phi
longitude: lamda
Earth radius: R
a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lamda/2)**2
c = 2 * arctan2(a**0.5, (1-a)**0.5)
d = R * c
"""

class Haversine:
    '''
    use the haversine class to calculate the distance between
    two lon/lat coordnate pairs.
    output distance available in kilometers, meters, miles, and feet.
    example usage: Haversine([lon1,lat1],[lon2,lat2]).feet
    '''
    def __init__(self,coord1,coord2):
        lon1,lat1 = coord1
        lon2,lat2 = coord2
        
        R=6371e3
        phi_1 = math.radians(lat1)
        phi_2 = math.radians(lat2)

        delta_phi = math.radians(lat2-lat1)
        delta_lambda = math.radians(lon2-lon1)

        a = math.sin(delta_phi/2.0)**2 + \
            math.cos(phi_1)*math.cos(phi_2) * \
            math.sin(delta_lambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        self.meters = R * c                    # meters
        self.km = self.meters / 1000.0         # kilometers
        self.miles = self.meters * 0.000621371 # miles
        self.feet = self.miles * 5280          # feet

if __name__ == "__main__":

    coord1 = (-122.4194, 37.7749)  # San Francisco, CA
    coord2 = (-118.2437, 34.0522)  # Los Angeles, CA

    distance = Haversine(coord1, coord2)

    print("distance meters:", distance.meters)
    print("distance kilometers:", distance.km)
    print("distance miles:", distance.miles)
    print("distance feet:", distance.feet)