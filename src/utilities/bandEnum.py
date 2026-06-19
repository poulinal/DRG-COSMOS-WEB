# AP 2026

import enum


class BandEnum(enum.Enum):
    F115W = "f115w"
    F150W = "f150w"
    F277W = "f277w"
    F444W = "f444w"
    F770W = "f770w"
    
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    
    def __format__(self, format_spec):
        return self.value.__format__(format_spec)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BandEnum):
            return self.value == other.value
        else:
            return NotImplemented
        
    def __hash__(self):
        return hash(self.value)
    
    def __lt__(self, other):
        if isinstance(other, BandEnum):
            return self.value < other.value
        else:
            return NotImplemented
        
    @staticmethod
    def get_band_from_string(s):
        for band in BandEnum:
            if band.value == s:
                return band
        raise ValueError(f"Invalid band string: {s}")
    
    @staticmethod
    def get_all_bands():
        return list(BandEnum)