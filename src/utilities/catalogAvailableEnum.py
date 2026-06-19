# AP 2026

import enum

class CatalogAvailableEnum(enum.Enum):
    COSMOS_WEB = "COSMOS_WEB"
    JADES = "JADES"
    MIGHTEE = "MIGHTEE"
    CHAMPS = "CHAMPS"
    S_BAND = "S_BAND"
    L_BAND = "L_BAND"
    SUPER_DEBLENDED = "SUPER_DEBLENDED"
    VLA_3GHZ = "VLA_3GHZ"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    
    def __format__(self, format_spec):
        return self.value.__format__(format_spec)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, CatalogAvailableEnum):
            return self.value == other.value
        else:
            return NotImplemented
        
    def __hash__(self):
        return hash(self.value)
    
    def __lt__(self, other):
        if isinstance(other, CatalogAvailableEnum):
            return self.value < other.value
        else:
            return NotImplemented
        
    @staticmethod
    def get_catalog_from_string(s):
        for catalog in CatalogAvailableEnum:
            if catalog.value == s:
                return catalog
        raise ValueError(f"Invalid catalog string: {s}")
    
    @staticmethod
    def get_all_catalogs():
        return list(CatalogAvailableEnum)