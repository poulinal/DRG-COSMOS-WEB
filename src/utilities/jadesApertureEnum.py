# AP 2026

#all aperature in JADES catalog DR5

import enum

class JadesApertureEnum(enum.Enum):
    """
    Enum that defines the different apertures used in the JADES catalog DR5. The values correspond to the index of the aperture in the catalog. For example, APER_0p25 corresponds to the aperture with a radius of 0.25 arcseconds, which is the 3rd aperture in the catalog (index 2). Note APER0 corresponds to the aperature radius for each filter that encircles 80% of the PSF flux.

    Args:
        enum (_type_): _description_
    """
    APER0 = 0
    APER_0p1 = 1
    APER_0p15 = 2
    APER_0p25 = 3
    APER_0p3 = 4
    APER_0p35 = 5
    APER_0p5 = 6
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __format__(self, format_spec):
        return self.name.__format__(format_spec)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, JadesApertureEnum):
            return self.value == other.value
        else:
            return NotImplemented
        
    def __hash__(self):
        return hash(self.value)
    
    
    def __lt__(self, other):
        if isinstance(other, JadesApertureEnum):
            return self.value < other.value
        else:
            return NotImplemented
        
    @staticmethod
    def get_aperture_from_string(s):
        for aperture in JadesApertureEnum:
            if aperture.name == s:
                return aperture
        raise ValueError(f"Invalid aperture string: {s}")
    
    @staticmethod
    def get_all_apertures():
        return list(JadesApertureEnum)