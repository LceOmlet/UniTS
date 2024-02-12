from ..registry import TEST_MODULE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import Ridge
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score    
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

@TEST_MODULE.register("kmeans")
class KmeanModule:
    def __init__(self, **kwargs):
        pass
    
    def evaluate(self, repr, label, per_batch, **kwargs):
        label_num = np.max(label) + 1
        pca = PCA(n_components=10)
        reps = pca.fit_transform(repr)
        kmeans = KMeans(label_num)
        pred = kmeans.fit_predict(reps)
        NMI_score = normalized_mutual_info_score(label, pred)
        RI_score = rand_score(label, pred)
        per_batch["clustering_rst"] = pred
        return {"NMI":NMI_score, "RI": RI_score}
    
    @staticmethod
    def collate(model, X, **kwargs):
        return {
            "repr": model.encode(X, **kwargs)
        }


@TEST_MODULE.register("svm")
class SVMModule:
    def __init__(self, repr, label, kernel="rbf", gamma='scale', search=False, **kwargs):
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        acc_val = -1
        C_best = None    
        for C in [10 ** i for i in range(-4, 5)]:
            clf = SVC(C=C, random_state=42)
            acc_i = cross_val_score(clf, repr, label, cv=5,)
            if acc_i.mean() > acc_val:
                C_best = C
        self.svc = SVC(kernel=kernel, gamma=gamma, C=C_best)
        
        self.svc.fit(repr, label)
    
    def evaluate(self, repr, label, **kwargs):
        # scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        pred = self.svc.predict(repr)
        report = classification_report(pred, label)
        score = accuracy_score(pred, label)
        return {
            "report": report,
            "accuracy": score 
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        rst = {
            "repr": model.encode(X, **kwargs)
        }
        return rst
        


@TEST_MODULE.register("logistic_regression")
class LRModule:
    def __init__(self, repr, label, **kwargs):
        self.lr = LogisticRegression()
        self.lr.fit(repr, label)
    
    def evaluate(self, repr, label, **kwargs):
        pred = self.lr.predict(repr)
        report = classification_report(pred, label)
        score = accuracy_score(pred, label)
        return {
            "report": report,
            "accuracy": score 
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        return {
            "repr": model.encode(X, **kwargs)
        }

@TEST_MODULE.register("ridge")
class RidgeModule:
    def __init__(self, repr, target, mask, valid_ratio, loss_module, **kwargs):
        alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        valid_split = int(len(repr) * valid_ratio)
        valid_repr, train_repr = repr[:valid_split], repr[valid_split:]
        valid_targets, train_targets = target[: valid_split], target[valid_split:]
        valid_masks, train_masks = mask[:valid_split], mask[valid_split :] 
        valid_results = []
        for alpha in alphas:
            target_shape = train_targets.shape[1:]
            lr = Ridge(alpha=alpha).fit(
                train_repr.reshape(train_repr.shape[0], -1), 
                train_targets.reshape(train_repr.shape[0], -1)
            )
            valid_pred = lr.predict(valid_repr.reshape((valid_repr.shape[0], -1)))
            valid_pred = valid_pred.reshape((valid_split, target_shape[0], target_shape[1]))
            score = loss_module(torch.tensor(valid_targets), torch.tensor(valid_pred), torch.tensor(valid_masks)).detach().cpu().numpy()
            score = np.mean(score)
            valid_results.append(score)
        best_alpha = alphas[np.argmin(valid_results)]
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(repr.reshape((repr.shape[0], -1)), target.reshape((repr.shape[0], -1)))
        self.ridge = ridge
        self.loss_module = loss_module
    
    def evaluate(self, repr, target, val_loss_module, mask, **kwargs):
        pred = self.ridge.predict(X=repr.reshape((repr.shape[0], -1)))
        pred = pred.reshape(target.shape)
        loss = val_loss_module(torch.tensor(target), torch.tensor(pred), torch.tensor(mask)).detach().cpu().numpy().mean()
        return {
            "loss": float(loss)
        }
    
    @staticmethod
    def collate(model, X, mask, **kwargs):
        # X = X.detach().clone()
        target= X
        X[mask] = 0
        kwargs["padding_masks"] = mask
        return {
            "repr": model.encode(X, **kwargs),
            "target": target,
        }
