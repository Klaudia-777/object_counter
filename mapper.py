import os
import re

directory = "test"
tansformer = {0: 6, 1: 7, 2: 80, 3: 5}
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        f = open(directory + "/" + filename, "r")
        lines = f.readlines()
        f.close()
        text = ""
        for line in lines:
            match = re.findall("^\d+", line)
            if len(match) > 0:
                current_number = match[0]
                if int(current_number) in tansformer:
                    new_line = re.sub("^\d+", str(tansformer[int(current_number)]), line)
                    text += new_line
                else:
                    text += line
            else:
                text += line
        f = open(directory + "/" + filename, "w")
        f.write(text)
        f.close()
    else:
        continue
