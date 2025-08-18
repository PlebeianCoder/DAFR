import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# CosineStat relies on their being an equal number of images everywhere

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_subdirectories_named(a_dir, tarName):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and tarName in name]

def makeBaselineStats(benchDir):
    total, imps, = 0, [0,0]
    utotal, uimps = 0, [0,0]
    coses = [0,0]
    # 0 index for 100, 1 for 1000 index
    
    allJson = "/Baseline/statistics/all.json"
    uniFalseJson = "/Baseline/statistics/universal_example_False.json"
    checked = 0
    for d in get_immediate_subdirectories(benchDir):
        # print(benchDir+"/"+d+allJson)
        j = json.load(open(benchDir+"/"+d+allJson))
        total += int(j["total"])
        imps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
        imps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
        coses[0] += float(j["CosineStat"]["mean"])

        j = json.load(open(benchDir+"/"+d+uniFalseJson))
        utotal += int(j["total"])
        uimps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
        uimps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
        coses[1] += float(j["CosineStat"]["mean"])
        checked += 1

    return  [s / total for s in imps], [s / utotal for s in uimps], total, utotal, [s/checked for s in coses]

def makeMatchStats(benchDir):
    total, imps, = 0, [0,0]
    utotal, uimps = 0, [0,0]
    coses = [0,0]
    # 0 index for 100, 1 for 1000 index
    
    allJson = "/MatchLighting/statistics/all.json"
    uniFalseJson = "/MatchLighting/statistics/universal_example_False.json"
    checked = 0
    for d in get_immediate_subdirectories(benchDir):
        # print(benchDir+"/"+d+allJson)
        j = json.load(open(benchDir+"/"+d+allJson))
        total += int(j["total"])
        imps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
        imps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
        coses[0] += float(j["CosineStat"]["mean"])
    
        j = json.load(open(benchDir+"/"+d+uniFalseJson))
        utotal += int(j["total"])
        uimps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
        uimps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
        coses[1] += float(j["CosineStat"]["mean"])
        checked += 1

    return  [s / total for s in imps], [s / utotal for s in uimps], total, utotal, [s/checked for s in coses]

# def makeBaselineNamedStats(benchDir, name):
    
#     utotal, uimps = 0,[0,0]
    
#     uniFalseJson = "/Baseline/statistics/universal_example_False.json"
#     for d in get_immediate_subdirectories_named(benchDir, name):
    
#         j = json.load(open(benchDir+"/"+d+uniFalseJson))
#         utotal += int(j["total"])
#         uimps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
#         uimps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])

#     return [s / utotal for s in uimps]

def calcAngled(benchDir):
    strTotal, strImps = 0,[0,0]
    angTotal, angImps = 0,[0,0]
    coses, ctotals = [0,0], [0,0]
    
    jsonStr = []
    jsonAng = []
    strAngles = ["0", "10", "-10"]
    for a in strAngles:
        for b in strAngles:
            jsonStr.append(f"/Baseline/statistics/universal_example_False-angle_pitch_{a}-angle_yaw_{b}.json")
            jsonStr.append(f"/Baseline/statistics/angle_pitch_{a}-angle_yaw_{b}-universal_example_False.json")
  
    angAngles = ["0", "10", "-10", "-20", "20", "30", "-30", "-40", "40"]
    for a in angAngles:
        for b in angAngles:
            if not (a in strAngles and b in strAngles): 
                jsonAng.append(f"/Baseline/statistics/universal_example_False-angle_pitch_{a}-angle_yaw_{b}.json")
                jsonAng.append(f"/Baseline/statistics/angle_pitch_{a}-angle_yaw_{b}-universal_example_False.json")

    for d in get_immediate_subdirectories(benchDir):
        for strJ in jsonStr:
            # print(benchDir+"/"+d+strJ)
            # print(benchDir+"/"+d+allJson)
            # print(benchDir+"/"+d+strJ)
            if os.path.exists(benchDir+"/"+d+strJ):
                # print(benchDir+"/"+d+strJ)
                j = json.load(open(benchDir+"/"+d+strJ))
                strTotal += int(j["total"])
                strImps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
                strImps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
                coses[0] += float(j["CosineStat"]["mean"])
                ctotals[0] += 1


    for d in get_immediate_subdirectories(benchDir):
        for angJ in jsonAng:
            # print(benchDir+"/"+d+allJson)
            if os.path.exists(benchDir+"/"+d+angJ):
                j = json.load(open(benchDir+"/"+d+angJ))
                angTotal += int(j["total"])
                angImps[0] += int(j["CosineSuccessRate"]["cosine_100_succ"])
                angImps[1] += int(j["CosineSuccessRate"]["cosine_1000_succ"])
                coses[1] += float(j["CosineStat"]["mean"])
                ctotals[1] += 1

    if strTotal == 0:
        return 0,0,0,0

    return [s / strTotal for s in strImps], [a / angTotal for a in angImps], strTotal, angTotal, [coses[i]/ctotals[i] for i in range(len(ctotals))]

def getStats(benchDir):
    # sr is straight, ar is angled
    sr, ar, stotal, atotal, coses = calcAngled(benchDir+"/benchmarks")
    # mar is all, mur is uni false
    mar, mur, matotal, mutotal, mcoses = makeMatchStats(benchDir+"/benchmarks")

    return f"For {benchDir}:\nStraight: {sr}, {stotal}\nAngled: {ar}, {atotal}\nML uni false: {mur}, {mutotal}\nCoses: {coses}, {mcoses}\n#############"

def get_directories_with_prefix(root_dir, prefix):
    return [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith(prefix)]

def getStyleStats(overallBench, benchPhrase):
    bdirs = get_immediate_subdirectories_named(overallBench, benchPhrase)
    all_means, all_s100, all_s1000 = [], [], []
    for b in bdirs:
        curDir = overallBench + b +"/benchmarks"
        print(b)
        print(curDir)
        allS, uniFalse, _, _, cs = makeBaselineStats(curDir)
        all_means.append(cs[1])
        all_s100.append(uniFalse[0])
        all_s1000.append(uniFalse[1])
        print(f"For {b}: {cs[1]}, {uniFalse[0]}, {uniFalse[0]}")
    
    print(f"{overallBench}: {sum(all_means)/len(all_means)}, {sum(all_s100)/len(all_s100)}, {sum(all_s1000)/len(all_s1000)}")

bdirs = []

# benchmarks.md on angled defined as less than 40 (including straights)
# new_benchmarks.md has a straight and angled disjoint
with open("new_benchStats.md", 'a') as file:
    for b in bdirs:
        print(f"Starting {b}")
        file.write(getStats(b)+"\n")

# for b in be:
#     getStyleStats(b, "sty_mfn")