import re

INPUT_FILE = "./dataset/test_labels.txt"
OUTPUT_FILE = "./dataset/test_labels_fa.txt"

mapping = {
    "A": "ا",
    "B": "ب",
    "P": "پ",
    "T": "ت",
    "TH": "ث",
    "J": "ج",
    "D": "د",
    "Z": "ز",
    "ZH": "ژ",
    "SIN": "س",
    "SH": "ش",
    "SAD": "ص",
    "TA": "ط",
    "EIN": "ع",
    "Q": "ق",
    "L": "ل",
    "M": "م",
    "N": "ن",
    "V": "و",
    "H": "ه",
    "Y": "ی"
}

keys_sorted = sorted(mapping.keys(), key=len, reverse=True)

def convert_label(label):
    i = 0
    out = ""
    while i < len(label):
        matched = False
        for k in keys_sorted:
            if label[i:i+len(k)] == k:
                out += mapping[k]
                i += len(k)
                matched = True
                break
        if not matched:
            out += label[i]
            i += 1
    return out


with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue

        try:
            path, label = line.rsplit(" ", 1)
        except ValueError:
            raise ValueError(f"Bad format in line {line_num}: {line}")
        new_label = convert_label(label)

        fout.write(f"{path} {new_label}\n")

print("Done. Converted labels saved to:", OUTPUT_FILE)
