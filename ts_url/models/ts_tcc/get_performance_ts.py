import json
datasets="ArticularyWordRecognition AtrialFibrillation BasicMotions Epilepsy ERing HandMovementDirection Libras NATOPS PEMS-SF PenDigits StandWalkJump UWaveGestureLibrary"
with open("eval.txt", "w") as eval:
    for data in datasets.split(" "):
        NMI = [data, "NMI"]
        RI = [" ", "RI"]
        for i in range(1, 6):
            with open(data + "_" + str(i) + ".txt", "r") as res:
                ans = json.load(res)
                NMI.append(str(ans["NMI"]))
                RI.append(str(ans["RI"]))
        eval.write(",".join(NMI) + "\n" + ','.join(RI) + "\n")