# AP 2026

import enum

class CosmosWebApertureEnum(enum.Enum):
    """
    Enum that defines the different apertures used in the Cosmos Web catalog. List: Aperture 0.2, 0.3, 0.5, 0.75, 1.0 arcseconds each corresponding to an index from 0 to 4. 

    Args:
        enum (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    APER_0p2 = 0
    APER_0p3 = 1
    APER_0p5 = 2
    APER_0p75 = 3
    APER_1p0 = 4
    
    def __str__(self):
        return f"aper_{self.value}"
    
    def __repr__(self):
        return f"aper_{self.value}"
    
    def __format__(self, format_spec):
        return f"aper_{self.value}".__format__(format_spec)
    
    @staticmethod
    def get_aperture_from_string(s):
        for aperture in CosmosWebApertureEnum:
            if f"aper_{aperture.value}" == s:
                return aperture
        raise ValueError(f"Invalid aperture string: {s}")
    
    @staticmethod
    def get_all_apertures():
        return list(CosmosWebApertureEnum)
    
    