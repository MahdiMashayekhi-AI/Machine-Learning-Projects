PLATE_MAP = {
    "ا": "A",
    "ب": "B",
    "پ": "P",
    "ت": "T",
    "ث": "S",
    "ج": "J",
    "د": "D",
    "ز": "Z",
    "ژ": "Zh",
    "س": "Sin",
    "ش": "Sh",
    "ص": "Sad",
    "ط": "Ta",
    "ع": "E",
    "ق": "G",
    "ل": "L",
    "م": "M",
    "ن": "N",
    "و": "V",
    "ه": "H",
    "ی": "Y"
}

def map_plate(plate):
    return "".join([PLATE_MAP.get(c, c) for c in plate])