import os
import glob
import torch
import random
import numpy as np
from pathlib import Path
from Diffv2Util import loadForLDM, l2_norm
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.clip import FaRL
from advfaceutil.recognition.iresnethead import IResNetHead
from insight2Adv import InsightNet
from advfaceutil.recognition.mobilefacenet import MobileFaceNet
import math
from sklearn.model_selection import KFold
from scipy import interpolate
from torch.nn import CosineSimilarity
import torchvision
from PIL import Image
from nn_modules import LandmarkExtractor, FaceXZooProjector
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


face_landmark_detector = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
sd = torch.load('./landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar',
                map_location=device)['state_dict']
face_landmark_detector.load_state_dict(sd)
location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)
uv_mask = torchvision.transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)

def maskIt(face_imgs, style_attack_mask):
    with torch.no_grad():
        # Need to resize to 112
        face_imgs = torch.unsqueeze(face_imgs, 0)
        to112 = torchvision.transforms.Resize(112)
        t_mask = to112(style_attack_mask)
        style_attack_mask = t_mask * uv_mask
        preds = location_extractor(face_imgs).to(device)
        style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
        style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)
        if style_masked_face.ndim > 3:
            style_masked_face = style_masked_face[0]
        return style_masked_face

# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py

class ThreshLoss(torch.nn.Module):
    def __init__(self, classes, samples):
        super(ThreshLoss, self).__init__()
        self.intraExamples = 19800
        self.interExamples = 182250

    def forward(self, intraCorrect, interCorrect):
        return (intraCorrect/self.intraExamples) + (interCorrect/self.interExamples)

def generate_unique_vectors(num_vectors, vector_size, c, l):
    # Generate a set of unique vectors
    unique_vectors = set()
    
    while len(unique_vectors) < num_vectors:
        # Create a random vector of the specified size
        vector = np.random.rand(vector_size)  # Use tuple to make it hashable
        # print(vector)
        vector[0] = int(math.floor(vector[0]*c))
        vector[1] = int(math.floor(vector[1]*l))
        vector[2] = int(math.floor(vector[2]*c))
        vector[3] = int(math.floor(vector[3]*l))
        vector = vector.astype(int)
        # print(vector)
        # print(c, l)
        if vector[0] == vector[2]:
            continue
        unique_vectors.add(tuple(vector))
    # print(unique_vectors)
    return list(unique_vectors)
        

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, paths, embedder, upperLimit, cache=False):
        with torch.no_grad():
            imgs = []
            labels = []
            for d in range(len(paths)):
                files = glob.glob(os.path.join(paths[d], '*.jpg')) + glob.glob(os.path.join(paths[d], '*.JPG'))
                # print(d)
                files = files[:upperLimit]
                # print(len(files))
                if cache:
                    for f in files:
                        rgbX, pilX = loadForLDM(str(f), device)
                        rgbX = torch.unsqueeze(rgbX, 0)
                        imgs.append(embedder.returnEmbedding(rgbX))
                else:
                    imgs = imgs + files
                labels = labels + [d for _ in range(upperLimit)]

            img_pairs = []
            label_pairs = []
            print(f"Got {len(imgs)}")
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    if cache:
                        img_pairs.append((i, j))
                    else:
                        img_pairs.append((imgs[i], imgs[j]))
                    label_pairs.append([labels[i], labels[j]])
                    

            self.img_pairs = img_pairs
            self.label_pairs = label_pairs
            self.embedder = embedder
            self.cache = cache
            if cache:
                self.imgs = imgs

    def __len__(self):
        return len(self.label_pairs)

    def __getitem__(self, idx):
        with torch.no_grad():
            if self.cache:
                embed1 = self.imgs[self.img_pairs[idx][0]]
                embed2 = self.imgs[self.img_pairs[idx][1]]
            else:
                rgbX, pilX = loadForLDM(str(self.img_pairs[idx][0]), device)
                rgbX = torch.unsqueeze(rgbX, 0)
                embed1 = self.embedder.returnEmbedding(rgbX)
                rgbX, pilX = loadForLDM(str(self.img_pairs[idx][1]), device)
                rgbX = torch.unsqueeze(rgbX, 0)
                embed2 = self.embedder.returnEmbedding(rgbX)
            label1, label2 = self.label_pairs[idx]
            return embed1, embed2, label1, label2

class EvenEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, paths, embedder, upperLimit, cache=False, masked=False):
        with torch.no_grad():
            intraExpect = len(paths) * (upperLimit * (upperLimit-1))/2
            interExpect = ((len(paths) * upperLimit) * ((len(paths) * upperLimit) - 1) / 2) - intraExpect
            random.seed(10)
            # print(intraExpect)
            # print(interExpect)
            self.interGood = random.sample(list(range(int(interExpect))), int(intraExpect))
            imgs = []
            labels = []
            for d in range(len(paths)):
                # print(f"Done {d}")
                files = glob.glob(os.path.join(paths[d], '*.jpg')) + glob.glob(os.path.join(paths[d], '*.JPG'))
                # print(paths[d])
                files = files[:upperLimit]
                # print(f"Count {len(files)}")
                if cache:
                    for f in files:
                        rgbX, pilX = loadForLDM(str(f), device)
                        rgbX = torch.unsqueeze(rgbX, 0)
                        if masked:
                            random_number = random.randint(1, 3)
                            if random_number == 1:
                                # black
                                face_mask = torch.ones((1, 3, 112, 112)).to(device) * 1e-6
                            elif random_number == 2:
                                # white
                                face_mask = torch.ones((1, 3, 112, 112)).to(device)
                            else:
                                # blue
                                face_mask = torch.zeros((1, 3, 112, 112)).to(device)
                                face_mask[0, 2] = 0.7
                            rgbX = torch.unsqueeze(maskIt(rgbX[0], face_mask),0)
                        imgs.append(embedder.returnEmbedding(rgbX))
                else:
                    imgs = imgs + files
                labels = labels + [d for _ in range(upperLimit)]

            img_pairs = []
            label_pairs = []
            foundInter = 0
            print(f"Got {len(imgs)}")
            

            for c in range(len(paths)):
                for i in range(upperLimit):
                    for j in range(i+1, upperLimit):
                        # print(((c*len(paths))+i, (c*len(paths))+j))
                        img_pairs.append(((c*upperLimit)+i, (c*upperLimit)+j))
                        label_pairs.append((c, c))

            # print(f"Intra {len(img_pairs)}")
            np.random.seed(10)
            v = generate_unique_vectors(int(intraExpect), 4, len(paths), upperLimit)
            
            for i in range(len(v)):
                # print(f"inter {i}")
                # print(((v[i][0]*upperLimit) + v[i][1], (v[i][2]*upperLimit) + v[i][3]))
                img_pairs.append(((v[i][0]*upperLimit) + v[i][1], (v[i][2]*upperLimit) + v[i][3]))
                label_pairs.append((labels[v[i][0]*upperLimit], labels[v[i][2]*upperLimit]))

                        

            self.img_pairs = img_pairs
            self.label_pairs = label_pairs
            self.embedder = embedder
            self.cache = cache
            if cache:
                self.imgs = imgs

    def __len__(self):
        return len(self.label_pairs)

    def __getitem__(self, idx):
        with torch.no_grad():
            if self.cache:
                # print(self.img_pairs[idx][0])
                # print(self.img_pairs[idx][1])
                embed1 = self.imgs[self.img_pairs[idx][0]]
                embed2 = self.imgs[self.img_pairs[idx][1]]
            else:
                rgbX, pilX = loadForLDM(str(self.img_pairs[idx][0]), device)
                rgbX = torch.unsqueeze(rgbX, 0)
                embed1 = self.embedder.returnEmbedding(rgbX)
                rgbX, pilX = loadForLDM(str(self.img_pairs[idx][1]), device)
                rgbX = torch.unsqueeze(rgbX, 0)
                embed2 = self.embedder.returnEmbedding(rgbX)
            label1, label2 = self.label_pairs[idx]
            return embed1, embed2, label1, label2



def getEmbeddings(paths, embedder, fillIn):
    #  names = [d[d.rfind("/") + 1:] for d in paths]
    for d in range(len(paths)):
        files = glob.glob(os.path.join(paths[d], '*.jpg')) + glob.glob(os.path.join(paths[d], '*.JPG'))
        files = files[:fillIn.size()[1]]
        for f in range(len(files)):
            rgbX, pilX = loadForLDM(str(files[f]), device)
            rgbX = torch.unsqueeze(rgbX, 0)
            fillIn[d][f] = embedder.returnEmbedding(rgbX)
            # already unit norm

    return fillIn


def calculateThresholdStats(paths, embedder, upperLimit):
    with torch.no_grad():
        names = [d[d.rfind("/") + 1:] for d in paths]
        print(names)
        allEmbeds = torch.zeros((len(names), upperLimit, 512)).to(device)
        allEmbeds = getEmbeddings(paths, embedder, allEmbeds)

        maxThresh, minThresh = 0, 100
        for i in range(allEmbeds.size()[0]):
            curDiffs = []
            for j in range(allEmbeds.size()[1]):
                for k in range(allEmbeds.size()[1]):
                    if j != k:
                        diff = torch.subtract(allEmbeds[i][j], allEmbeds[i][k])
                        target = torch.sum(torch.square(diff)).item()
                        curDiffs.append(target)

            minT, maxT = min(curDiffs), max(curDiffs)
            curDiffs = np.array(curDiffs)
            if maxThresh < maxT:
                maxThresh = maxT
            if minThresh > minT:
                minThresh = minT
            print(f"Thresh Stats for {names[i]}:\nMin: {minT}\nMax: {maxT}\nAvg: {np.mean(curDiffs)}\nStd: {np.std(curDiffs)}\n")

        print(f"Overall thresh: Min: {minThresh}, Max: {maxThresh}\n################### MOVING TO DIFFERENCES ###################")
        maxThresh, minThresh = 0, 100
        for a in range(allEmbeds.size()[0]):
            for i in range(a+1, allEmbeds.size()[0]):
                curDiffs = []
                for j in range(allEmbeds.size()[1]):
                    for k in range(allEmbeds.size()[1]):
                        if j != k:
                            diff = torch.subtract(allEmbeds[a][j], allEmbeds[i][k])
                            target = torch.sum(torch.square(diff)).item()
                            curDiffs.append(target)

                minT, maxT = min(curDiffs), max(curDiffs)
                curDiffs = np.array(curDiffs)
                if maxThresh < maxT:
                    maxThresh = maxT
                if minThresh > minT:
                    minThresh = minT
                print(f"Thresh Stats from {names[a]} to {names[i]}:\nMin: {minT}\nMax: {maxT}\nAvg: {np.mean(curDiffs)}\nStd: {np.std(curDiffs)}\n")

        print(f"Overall thresh: Min: {minThresh}, Max: {maxThresh}")
    
def optimiseThresh(paths, embedder, upperLimit, epochNum):
    allEmbeds = EvenEmbeddingDataset(paths, embedder, upperLimit, cache=True)
    print(len(allEmbeds))
    loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, shuffle=True)

    thresh = torch.tensor([1.2], dtype=torch.float32, requires_grad=True, device=device)
    adam = torch.optim.Adam([thresh], lr=1e-3, amsgrad=True)
    relu = torch.nn.ReLU()
    intraExpect = len(paths) * (upperLimit * (upperLimit-1))/2
    interExpect = ((len(paths) * upperLimit) * ((len(paths) * upperLimit) - 1) / 2) - intraExpect
    for epoch in range(epochNum):
        interCorrect, intraCorrect = 0,0
        interExamples, intraExamples = 0, 0
        epochLoss, iterCount, iterLoss = 0, 0, 0 
        for e1, e2, id1, id2 in loader:
            iterCount += 1
            if iterCount % 10000 == 0:
                print(f"Cur Thresh {(iterCount/len(allEmbeds)) * 100:.2f}%: {thresh.item()} and loss of {iterLoss:.4f}")
                iterLoss = 0
            diff = torch.sum(torch.square(torch.subtract(e1, e2)))
            if id1 != id2:
                interExamples += 1
                # Want inner to be negative: thresh < diff

                loss = relu(torch.subtract(thresh, diff))
                if loss.item() == 0:
                    interCorrect += 1
                    continue # as grad is 0
            else:
                intraExamples += 1
                # Want inner to be negative: thresh > diff

                loss = relu(torch.subtract(diff, thresh))
                if loss.item() == 0:
                    intraCorrect += 1
                    continue # as grad is 0

            
            adam.zero_grad()
            epochLoss += loss.item()
            iterLoss += loss.item()
            loss.backward()
            adam.step()


        # loss = threshLoss(intraCorrect, interCorrect)
        print(f"Epoxh {epoch}\nInter Rate: {interCorrect}/{interExamples}, {(interCorrect/interExamples) * 100}\nIntra Rate: {intraCorrect}/{intraExamples}, {(intraCorrect/intraExamples) * 100}")
        print(f"Loss: {epochLoss}")


        print(f"New Threshold: {thresh.item()}")

def testThresh(paths, embedder, upperLimit, thresh):
    with torch.no_grad():
        allEmbeds = EmbeddingDataset(paths, embedder, upperLimit, cache=True)
        print(len(allEmbeds))
        loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, shuffle=True)
        tp, fp, tn, fn = 0, 0, 0, 0
        intraExpect = len(paths) * (upperLimit * (upperLimit-1))/2
        interExpect = ((len(paths) * upperLimit) * ((len(paths) * upperLimit) - 1) / 2) - intraExpect
        summedError = 0

        interCorrect, intraCorrect = 0,0
        interExamples, intraExamples = 0, 0
        for e1, e2, id1, id2 in loader:
            diff = torch.sum(torch.square(torch.subtract(e1, e2)))
            diff = thresh-diff.item()
            if id1==id2:
                intraExamples+=1
                if diff >= 0:
                    intraCorrect+=1
                    tp+=1
                else:
                    fn+=1
                    summedError += abs(diff/intraExpect)
            else:
                interExamples+=1
                if diff < 0:
                    interCorrect+=1
                    tn+=1
                else:
                    fp+=1
                    summedError += abs(diff/interExpect)

        tpr, fpr, tnr, fnr = tp/intraExpect, fp/intraExpect, tn/interExpect, fn/interExpect
        print(f"Inter Rate: {interCorrect}/{interExamples}, {(interCorrect/interExamples) * 100}%\nIntra Rate: {intraCorrect}/{intraExamples}, {(intraCorrect/intraExamples) * 100}%")
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\nTPR: {tp/intraExpect}\nFPR: {fp/intraExpect}\nTNR: {tn/interExpect}\nFNR: {fn/interExpect}")   
        print(f"SUMMED ERROR: {summedError}")
        return summedError


def getThresh(dataset, size, celebDir, researchDir, wd, upperLimit, epochNum):
    dirPaths = []

    faceClasses = size.class_names

    for i in range(len(faceClasses)):
        if i >= len(faceClasses)-3:
            dirPaths.append(researchDir+faceClasses[i])
        elif i < 2397:
            dirPaths.append(celebDir+faceClasses[i])

    inet = IResNet.construct(
        dataset, size, training=False, device=device, weights_directory=wd
    )

    optimiseThresh(dirPaths, inet, upperLimit, epochNum)


# val is tpr, far is fpr
def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = torch.logical_not(torch.less(dist, threshold))
    true_accept = torch.sum(torch.logical_and(predict_issame, actual_issame))
    false_accept = torch.sum(
        torch.logical_and(predict_issame, torch.logical_not(actual_issame)))
    n_same = torch.sum(actual_issame)
    n_diff = torch.sum(torch.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def calculate_val(thresholds,
                  paths,
                  embedder,
                  upperLimit,
                  far_target,
                  masked=True,
                  nrof_folds=10):
    with torch.no_grad():
        allEmbeds = EvenEmbeddingDataset(paths, embedder, upperLimit, cache=True, masked=masked)
        nrof_pairs = len(allEmbeds)
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=True)

        val = torch.zeros(nrof_folds)
        far = torch.zeros(nrof_folds)

        # diff = torch.subtract(embeddings1, embeddings2)
        # dist = torch.sum(torch.square(diff), 1)
        indices = torch.arange(nrof_pairs)
        cos_sim = CosineSimilarity()
        
        final_thresholds = []
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print(f"On {fold_idx}")
            # Find the threshold that gives FAR = far_target
        
            loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_set))
            dist = torch.zeros((len(train_set))).to(device)
            actual_issame = torch.zeros((len(train_set))).to(device)
            i=0
            for e1, e2, id1, id2 in loader:
                # diff = torch.subtract(e1, e2)
                # dist[i] = torch.sum(torch.square(diff)).item()
                dist[i] = cos_sim(e1[0], e2[0])
                if id1==id2:
                    actual_issame[i] = 1
                i+=1
            far_train = torch.zeros(nrof_thresholds).to(device)

            for threshold_idx, threshold in enumerate(thresholds):
                # print(threshold)
                _, far_train[threshold_idx] = calculate_val_far(
                    threshold, dist, actual_issame)
            
            print(far_train)
            if torch.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train.cpu().numpy(), thresholds, kind='slinear')
                threshold = f(far_target)
                # print(threshold)
            else:
                threshold = 0.0
            print(far_train.size())
            print(far_train)
            loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_set))
            dist = torch.zeros((len(test_set))).to(device)
            actual_issame = torch.zeros((len(test_set))).to(device)
            i=0
            for e1, e2, id1, id2 in loader:
                # diff = torch.subtract(e1, e2)
                # dist[i] = torch.sum(torch.square(diff)).item()
                dist[i] = cos_sim(e1[0], e2[0])
                if id1==id2:
                    actual_issame[i] = 1
                i+=1

            print(threshold)
            final_thresholds.append(threshold)
            val[fold_idx], far[fold_idx] = calculate_val_far(
                float(threshold), dist, actual_issame)


        print(f"Mean threshold: {sum(final_thresholds)/len(final_thresholds)}")
        val_mean = torch.mean(val)
        far_mean = torch.mean(far)
        val_std = torch.std(val)
        return val_mean, val_std, far_mean


def calculateAccuracyVal(thresholds,
                  paths,
                  embedder,
                  upperLimit,
                  nrof_folds=10):
    # Find the best threshold for the fold
    with torch.no_grad():
        allEmbeds = EvenEmbeddingDataset(paths, embedder, upperLimit, cache=True)
        nrof_pairs = len(allEmbeds)
        nrof_thresholds = len(thresholds)
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=True)

        tprs = torch.zeros((nrof_folds, nrof_thresholds))
        fprs = torch.zeros((nrof_folds, nrof_thresholds))
        accuracy = torch.zeros((nrof_folds))
        indices = torch.arange(nrof_pairs)
        best_thresholds = []
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_set))
            dist = torch.zeros((len(train_set))).to(device)
            actual_issame = torch.zeros((len(train_set))).to(device)
            i=0
            for e1, e2, id1, id2 in loader:
                diff = torch.subtract(e1, e2)
                dist[i] = torch.sum(torch.square(diff)).item()
                if id1==id2:
                    actual_issame[i] = 1
                i+=1

            acc_train = torch.zeros((nrof_thresholds))
            train_tpr = torch.zeros((nrof_thresholds))
            train_fpr = torch.zeros((nrof_thresholds))
            train_tnr = torch.zeros((nrof_thresholds))
            train_f1 = torch.zeros((nrof_thresholds))
            train_prec = torch.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                train_tpr[threshold_idx], train_fpr[threshold_idx], train_tnr[threshold_idx], train_f1[threshold_idx], train_prec[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(
                    threshold, dist, actual_issame)
            best_threshold_index = torch.argmax(acc_train)
            loader = torch.utils.data.DataLoader(allEmbeds, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_set))
            dist = torch.zeros((len(test_set))).to(device)
            actual_issame = torch.zeros((len(test_set))).to(device)
            i=0
            for e1, e2, id1, id2 in loader:
                diff = torch.subtract(e1, e2)
                dist[i] = torch.sum(torch.square(diff)).item()
                if id1==id2:
                    actual_issame[i] = 1
                i+=1
            print(f"For fold {fold_idx}")
            # print(thresholds)
            # print(acc_train)
            print(list(zip(thresholds, acc_train.tolist())))
            print(list(zip(thresholds, train_tpr.tolist())))
            print(list(zip(thresholds, train_fpr.tolist())))
            print(list(zip(thresholds, train_tnr.tolist())))
            print(list(zip(thresholds, train_f1.tolist())))
            print(list(zip(thresholds, train_prec.tolist())))
            print('threshold', thresholds[best_threshold_index])
            best_thresholds.append(thresholds[best_threshold_index])
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx,
                        threshold_idx], fprs[fold_idx,
                                            threshold_idx], _, _, _, _ = calculate_accuracy(
                                                threshold, dist,
                                                actual_issame)
            _, _, _, _, _, accuracy[fold_idx] = calculate_accuracy(
                thresholds[best_threshold_index], dist,
                actual_issame)

        tpr = torch.mean(tprs, 0)
        fpr = torch.mean(fprs, 0)
        print(f"Best thresholds: {best_thresholds}")
        return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = torch.less(dist, threshold)
    tp = torch.sum(torch.logical_and(predict_issame, actual_issame))
    fp = torch.sum(torch.logical_and(predict_issame, torch.logical_not(actual_issame)))
    tn = torch.sum(
        torch.logical_and(torch.logical_not(predict_issame),
                       torch.logical_not(actual_issame)))
    fn = torch.sum(torch.logical_and(torch.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    tnr = 0 if (fp + tn == 0) else float(tn) / float(fp + tn)
    f1 = 0 if (2*tp + fn + fp == 0) else float(2*tp) / float(2*tp  +fn + fp)
    precision = 0 if (tp + fp== 0) else float(tp) / float(tp + fp)
    acc = float(tp + tn) / dist.size()[0]
    # acc = float(tpr + tnr) / 2
    return tpr, fpr, tnr, f1, precision, acc

celebDir = "celeb_test_dir"
researchDir = "res_test_dir"

dirPaths = []

for thr in [0.01]:
    for r in ["mfn"]:
        for s in ["LARGE", "FINAL"]:
            dataset = FaceDatasets["VGGFACE2"]
            size = dataset.get_size(s)
            faceClasses = size.class_names
            dirPaths = []

            for i in range(len(faceClasses)):
                if i >= len(faceClasses)-3:
                    dirPaths.append(researchDir+faceClasses[i])
                elif i < 997:
                    dirPaths.append(celebDir+faceClasses[i])
            print(f"Doing {r}")
            if r == "fted100":
                inet = IResNetHead.construct(
                    dataset, dataset.get_size("LARGE"), training=False, device=device, weights_directory=Path("weights_dir")
                )
            elif r == "farl":
                inet = FaRL.construct(
                    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
                    weights_directory=Path("weights_dir")
                )
            elif r == "mfn":
                inet = MobileFaceNet.construct(
                FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
                weights_directory=Path("weights_dir")
            )
            else:
                inet = IResNet.construct(
                    dataset, size, training=False, device=device, weights_directory=Path(f"weights_dir")
                )

            testThreshes = [s*0.001 for s in range(1000)]

            val_mean, val_std, far_mean = calculate_val(testThreshes, dirPaths, inet, 3, thr , masked=True)
            print(f"For {s} {r} and {thr}")

            print(val_mean)
            print(val_std)
            print(far_mean)