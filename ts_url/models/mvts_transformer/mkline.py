import json

with open("/home/liangchen/Desktop/mvts.txt", "r") as rst:
    mvts = json.load(rst)


res = ""
for key in mvts:
    nmi = ["NMI"]
    ri = ["RI"]
    for rec in mvts[key]:
        nmi.append(str(rec['NMI']))
        ri.append(str(rec['RI']))
    res += key + "," + ",".join(nmi) + "\n"
    res += ", " + ",".join(ri) + "\n"

with open("form_res.txt", "w") as fr:
    fr.write(res)


