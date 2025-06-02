import ephem
from datetime import datetime, timedelta

class MoonCalculator:
    def __init__(self):
        self.moon = ephem.Moon()
        
    def get_moon_phase(self, date):
        """Calculate moon phase for a given date"""
        self.moon.compute(date)
        # Returns value from 0-1 where 0.5 is full moon
        return self.moon.phase / 100.0
        
    def is_full_moon(self, date, threshold=0.95):
        """Check if given date is close to full moon"""
        phase = self.get_moon_phase(date)
        return phase >= threshold
        
    def find_full_moons(self, start_date, end_date):
        """Find all full moon dates between start_date and end_date"""
        full_moons = []
        current = ephem.Date(start_date)
        end = ephem.Date(end_date)
        
        while current < end:
            next_full = ephem.next_full_moon(current)
            if next_full > end:
                break
                
            full_moons.append(ephem.Date(next_full).datetime())
            current = ephem.Date(next_full + 1)
            
        return full_moons
        
    def get_moon_distance(self, date):
        """Calculate moon's distance from Earth in kilometers"""
        self.moon.compute(date)
        return self.moon.earth_distance * 149597870.691  # Convert AU to km

    def get_moon_illumination(self, date):
        """Get percentage of moon illumination"""
        self.moon.compute(date)
        return self.moon.phase 