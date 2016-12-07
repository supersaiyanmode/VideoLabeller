import sys
import re

string = re.sub("\s+", " ", ",".join(x.strip() for x in sys.stdin))
string = string.replace("  ", ",").replace("[ ","[").replace(" ", ",")
arr = eval(string)
with open("/tmp/acc.txt", "w") as f:
    print >>f, "A, B, C, D, E, F"
    for row in arr:
        print >>f, ", ".join(map(str, row))
print float(sum(arr[i][i] for i in range(len(arr)))) / sum(sum(arr, []))
