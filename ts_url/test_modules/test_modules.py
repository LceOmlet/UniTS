from ..registry import TEST_METHODS, TRANSFORMATION
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import Ridge
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score    
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import rand_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils import shuffle

SAMPLED_INDICES = None  # 存储采样数据的索引

def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=int)
    new_label[idx] = label

    return new_label

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    # print(np.sum(predict))
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    pred_entities = 0
    label_entities = 0

    for sp in splits:
        if is_anomaly:
            anomaly_pred_entity = np.sum(predict[pos: max(pos + delay + 1, sp)])
            if anomaly_pred_entity:
                pred_entities += anomaly_pred_entity
                assert sp > pos
                label_entities += sp - pos
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: max(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0
    # print(np.sum(new_predict))
    f_scale = label_entities / max(pred_entities , 1)
    return new_predict, f_scale


def regulize_anom_label_pred(predict, label, delay=7):
    test_timestamps = range(len(label))
    label = reconstruct_label(test_timestamps, label)
    predict = reconstruct_label(test_timestamps, predict)
    predict, f_scale = get_range_proba(predict, label, delay)
    label, predict = label.astype(bool), predict.astype(bool)
    tp = np.sum(label & predict)
    fp = np.sum((~label) & predict)
    fn = np.sum(label & (~predict))
    tn = np.sum((~label) & (~predict))
    # print(tn)
    # print(predict == 0 )
    precision = tp / (tp + fp * f_scale + 1e-7)
    # if tp == 0:
    #     exit()
    recall = tp / (tp + fn + 1e-7)
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)
    acc = (tp + tn) / (tp + tn + fn + fp * f_scale)
    # print(tp, tn, fp, fn)
    return f1, precision, recall, acc, predict, label


def moving_mean(arr, window):
    """
    计算移动平均值，使用NumPy的卷积函数，并在两端进行边缘填充。
    参数:
    arr: 输入数组
    window: 滑动窗口的大小

    返回:
    移动平均值的数组
    """
    # 创建一个等于窗口大小的、值为1/window的数组，作为卷积核
    kernel = np.ones(window) / window
    # 边缘填充，填充长度为(window-1)//2，以保持输出长度与输入长度相同
    pad_width = (window - 1) // 2
    padded_arr = np.pad(arr, pad_width, mode='edge')
    # 使用np.convolve计算卷积，'same'模式以保持输出和输入长度相同
    return np.convolve(padded_arr, kernel, mode='same')[pad_width:-pad_width]

DELAY = 0
@TEST_METHODS.register("isolation")
class IsolationForest_:
    def __init__(self, repr, **kwargs) -> None:
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        
    def calculate_vus_roc(self, scores, labels, window_size=100):
        """Calculate VUS-ROC (Volume Under the Surface ROC) metric.
        
        Args:
            scores: Anomaly scores
            labels: Ground truth labels
            window_size: Size of sliding window for calculating ROC curves
            
        Returns:
            VUS-ROC score
        """
        # Check for NaN values
        if np.isnan(scores).any() or np.isnan(labels).any():
            print("Warning: NaN values detected in scores or labels")
            return np.nan
            
        n_windows = len(scores) - window_size + 1
        if n_windows <= 0:
            print("Warning: Window size too large for the data")
            return np.nan
            
        vus_roc = 0
        valid_windows = 0
        
        for i in range(n_windows):
            window_scores = scores[i:i+window_size]
            window_labels = labels[i:i+window_size]
            
            # Skip windows with only one class
            if len(np.unique(window_labels)) < 2:
                continue
                
            try:
                # Calculate ROC curve for this window
                fpr, tpr, _ = roc_curve(window_labels, window_scores)
                if len(fpr) > 1 and len(tpr) > 1:  # Ensure we have enough points
                    vus_roc += auc(fpr, tpr)
                    valid_windows += 1
            except Exception as e:
                print(f"Warning: Error calculating ROC for window {i}: {str(e)}")
                continue
        
        if valid_windows == 0:
            print("Warning: No valid windows for ROC calculation")
            return np.nan
            
        return vus_roc / valid_windows
    
    def calculate_vus_pr(self, scores, labels, window_size=100):
        """Calculate VUS-PR (Volume Under the Surface PR) metric.
        
        Args:
            scores: Anomaly scores
            labels: Ground truth labels
            window_size: Size of sliding window for calculating PR curves
            
        Returns:
            VUS-PR score
        """
        n_windows = len(scores) - window_size + 1
        vus_pr = 0
        
        for i in range(n_windows):
            window_scores = scores[i:i+window_size]
            window_labels = labels[i:i+window_size]
            
            # Calculate PR curve for this window
            precision, recall, _ = precision_recall_curve(window_labels, window_scores)
            vus_pr += auc(recall, precision)
            
        return vus_pr / n_windows

    def evaluate(self, repr, label, per_batch, **kwargs):
        # repr = self.scaler.fit_transform(repr)
        conts = [0.001 + 0.002*i for i in range(10)]
        best_f1 = 0
        best_acc = 0
        best_cc = 0
        best_scores = None
        for cc in conts:
            self.isolat = IsolationForest(contamination=cc)
            self.isolat.fit(repr)  # First fit the model
            scores = -self.isolat.score_samples(repr)  # Get continuous anomaly scores
            pred = scores > np.percentile(scores, (1-cc)*100)  # Convert to binary predictions
            for delay in range(1, 20):
                f1, precision, recall, acc, predict, label = regulize_anom_label_pred(pred, label, delay=delay)
                if f1 >= best_f1:
                    best_f1 = f1
                    best_acc = acc
                    best_cc = cc
                    best_predict = predict
                    best_scores = scores
                    DELAY = delay
                    best_precision = precision
                    best_recall = recall
        
        # Calculate additional metrics
        auc_roc = roc_auc_score(label, best_scores)
        vus_roc = self.calculate_vus_roc(best_scores, label)
        vus_pr = self.calculate_vus_pr(best_scores, label)
        
        print(f"best delay: {DELAY}")
        print(f"best cc: {best_cc}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"VUS-ROC: {vus_roc:.4f}")
        print(f"VUS-PR: {vus_pr:.4f}")
        
        return {
            "f1": best_f1,
            "accuracy": best_acc,
            "precision": best_precision,
            "recall": best_recall,
            "predict": best_predict,
            "auc_roc": auc_roc,
            "vus_roc": vus_roc,
            "vus_pr": vus_pr
        }
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        return {
            "repr": model.encode(X, **kwargs)
        }

def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@TEST_METHODS.register("loss_ad")
class AlignmentLossScore:
    def __init__(self, repr, **kwargs) -> None:
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        

    def evaluate(self, repr, repr_aug, label, per_batch, **kwargs):
        # repr = self.scaler.fit_transform(repr)
        test_err = np.abs(repr - repr_aug).sum(axis=1)
        ma = moving_mean(test_err, 21)
        test_err_adj = (test_err - ma) / ma


        thr = np.mean(test_err_adj) + 4 * np.std(test_err_adj)
        pred = (test_err_adj > thr) * 1

        print(f"total anoms: {sum(label)}")
        print(f"total pred: {sum(pred)}")
        # print(label.shape)
        # raise RuntimeError()
        print(repr.shape)

        f1 = f1_score(pred, label)
        acc = accuracy_score(pred, label)

        return {
            "F1": f1,
            "accuracy": acc
        }
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        return {
            "repr": model.encode(X, **kwargs),
            "repr_aug": model.encode(
                X + torch.randn(X.shape).to(X.device) * 1e-3, **kwargs
            )
        }

def average_dtw(series_group1, series_group2):
    """
    计算两组 n 个 d 维时间序列的平均 DTW 距离。

    Args:
        series_group1 (list of np.ndarray): 第一组 n 个 d 维时间序列，每个元素是一个 (T1, d) 形状的时间序列
        series_group2 (list of np.ndarray): 第二组 n 个 d 维时间序列，每个元素是一个 (T2, d) 形状的时间序列

    Returns:
        float: 两组时间序列的平均 DTW 距离
    """
    # 检查输入长度是否一致
    if len(series_group1) != len(series_group2):
        raise ValueError("两组时间序列数量不一致")
    
    n = len(series_group1)  # 序列组的数量
    total_dtw = 0.0
    
    for seq1, seq2 in zip(series_group1, series_group2):
        # 检查每个时间序列的维度是否一致
        if seq1.shape[0] != seq2.shape[0]:
            raise ValueError("对应时间序列的维度不一致")
        
        # 计算当前 d 维时间序列的平均 DTW 距离
        d = seq1.shape[0]
        avg_dtw = 0.0
        
        for i in range(d):
            # 对每个维度单独计算 DTW 距离
            # print(seq1[i, :].shape, seq2[i, :].shape, seq1[i,:].ndim)
            distance, _ = fastdtw(seq1[i, :], seq2[i, :], dist=2)
            # distance = np.mean((seq1[i, :]- seq2[i, :]) ** 2)
            avg_dtw += distance / len(seq1[i, :])
        
        # 取每个 d 维时间序列的平均 DTW 距离
        avg_dtw /= d
        total_dtw += avg_dtw
    
    # 返回所有时间序列的平均 DTW 距离
    return total_dtw / n

@TEST_METHODS.register("time_vae")
class AlignmentLossScore:
    def __init__(self, X, X_rec, **kwargs) -> None:
        self.train_repr = repr
        self.train_X = X
        self.train_X_rec = X_rec
        
    def evaluate(self, X, X_rec, **kwargs):
        rec_std = np.std(X_rec)
        train_rec_std = np.std(self.train_X_rec)
        std_X = np.std(X, axis=-1)
        std_rec_X = np.std(X_rec, axis=-1)
        X_rec = X_rec * (std_X / std_rec_X)[..., None]
        std_X = np.std(self.train_X, axis=-1)
        std_rec_X = np.std(self.train_X_rec, axis=-1)
        train_X_rec = self.train_X_rec * (std_X / std_rec_X)[..., None]
        test_dtw = average_dtw(X, X_rec)
        return {
            "train_rec_loss": np.mean((self.train_X - train_X_rec) ** 2),
            "test_rec_loss": np.mean((X - X_rec) ** 2),
            "rec_std": rec_std,
            "train_rec_std": train_rec_std,
            "test_dtw": test_dtw
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        X_ = X.permute(0, 2, 1)
        z_mean, z_log_var, z = model.encoder(X_)
        repr = AlignmentLossScore.sample(z_mean, z_log_var, 4)
        # repr = z_mean + torch.rand_like(z_mean, device=z_mean.device, dtype=z_mean.dtype) * 0.8
        # print(repr - z_mean)
        X_rec = model.decode(repr).permute(0, 2, 1) 
        X_rec = X_rec 
        return {
            "repr": z_mean,
            "log_var": z_log_var,
            "X": X,
            "X_rec": X_rec
        }
        
    @staticmethod
    def sample(z_mean, z_log_var, scale=1):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device) * scale
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
@TEST_METHODS.register("spec")
class KmeanModule:
    def __init__(self, **kwargs):
        pass
    
    def evaluate(self, repr, label, per_batch, **kwargs):
        label_num = len(set(label)) 
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        # print(repr.shape)
        # raise RuntimeError()
        pca = PCA(n_components=10)
        reps = pca.fit_transform(repr)
        kmeans = SpectralClustering(label_num)
        pred = kmeans.fit_predict(reps)
        NMI_score = normalized_mutual_info_score(label, pred)
        RI_score = rand_score(label, pred)
        per_batch["clustering_rst"] = pred
        return {"NMI":NMI_score, "RI": RI_score}
    
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        return {
            "repr": model.encode(X, **kwargs)
        }


@TEST_METHODS.register("kmeans")
class KmeanModule:
    def __init__(self, **kwargs):
        pass
    
    def evaluate(self, repr, label, per_batch, **kwargs):
        label_num = len(set(label)) 
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        # print(repr.shape)
        # raise RuntimeError()
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
        kwargs.pop("mask", None)
        return {
            "repr": model.encode(X, **kwargs)
        }

@TEST_METHODS.register("gbdt")
class GBDT:
    def __init__(self, repr, label, **kwargs) -> None:
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        self.gbdt = GradientBoostingClassifier().fit(repr, label)

    def evaluate(self, repr, label, **kwargs):
        repr = self.scaler.fit_transform(repr)
        pred = self.gbdt.predict(repr)
        report = classification_report(pred, label)
        score = accuracy_score(pred, label)
        return {
            "report": report,
            "accuracy": score
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        rst = {
            "repr": model.encode(X, **kwargs)
        }
        return rst
    
@TEST_METHODS.register("hgbdt")
class HGBDT:
    def __init__(self, repr, label, **kwargs) -> None:
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        self.gbdt = HistGradientBoostingClassifier().fit(repr, label)

    def evaluate(self, repr, label, **kwargs):
        repr = self.scaler.fit_transform(repr)
        pred = self.gbdt.predict(repr)
        report = classification_report(pred, label)
        score = accuracy_score(pred, label)
        return {
            "report": report,
            "accuracy": score
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        rst = {
            "repr": model.encode(X, **kwargs)
        }
        return rst
    
@TEST_METHODS.register("svm")
class SVMModule:
    def __init__(self, repr, label, sample_rate=0.15, kernel="rbf", gamma='scale', search=False, **kwargs):
        global SAMPLED_INDICES
        
        self.scaler = RobustScaler()
        repr = self.scaler.fit_transform(repr)
        
        # 检查是否已经有采样索引，如果没有则计算
        if SAMPLED_INDICES is None:
            # 按类别均匀采样数据
            label = np.array(label)
            unique_labels = np.unique(label)
            total_samples = int(len(repr) * sample_rate)
            samples_per_class = total_samples // len(unique_labels)
            
            sampled_indices = []
            for class_label in unique_labels:
                class_indices = np.where(label == class_label)[0]
                # 如果该类别样本数少于需要采样的数量，则全部选择
                if len(class_indices) <= samples_per_class:
                    sampled_indices.extend(class_indices)
                else:
                    # 随机选择该类别的samples_per_class个样本
                    np.random.seed(42)  # 设置随机种子保证可重现性
                    selected_indices = np.random.choice(
                        class_indices, 
                        size=samples_per_class, 
                        replace=False
                    )
                    sampled_indices.extend(selected_indices)
            
            # 随机打乱采样后的索引
            sampled_indices = np.array(sampled_indices)
            np.random.seed(42)
            np.random.shuffle(sampled_indices)
            
            # 存储采样索引到全局变量
            SAMPLED_INDICES = sampled_indices
            
            print(f"Generated new sampling indices: {len(sampled_indices)} samples from {len(repr)} total samples")
            print(f"Samples per class: {dict(zip(unique_labels, [np.sum(np.array(label)[sampled_indices] == l) for l in unique_labels]))}")
        else:
            print(f"Using existing sampling indices: {len(SAMPLED_INDICES)} samples")
        
        # 使用已存储的采样索引
        repr_sampled = repr[SAMPLED_INDICES]
        label_sampled = np.array(label)[SAMPLED_INDICES]
        
        acc_val = -1
        C_best = None    
        for C in [10 ** i for i in range(-4, 5)]:
            clf = SVC(C=C, random_state=42)
            try:
                acc_i = cross_val_score(clf, repr_sampled, label_sampled, cv=4)
            except Exception as e:
                acc_i = np.array([0.0, 0.0, 0.0, 0.0])
            if acc_i.mean() > acc_val:
                acc_val = acc_i.mean()
                C_best = C
        
        self.svc = SVC(kernel=kernel, gamma=gamma, C=C_best)
        self.svc.fit(repr_sampled, label_sampled)
    
    def evaluate(self, repr, label, **kwargs):
        # scaler = RobustScaler()
        repr = self.scaler.transform(repr)
        pred = self.svc.predict(repr)
        # raise RuntimeError()
        report = classification_report(pred, label)
        score = accuracy_score(pred, label)
        
        return {
            "report": report,
            "accuracy": score 
        }
    
    @staticmethod
    def collate(model, X, **kwargs):
        kwargs.pop("mask", None)
        reprs = model.encode(X, **kwargs)
        rst = {
            "repr": reprs
        }
        return rst
        


@TEST_METHODS.register("logistic_regression")
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
        kwargs.pop("mask")
        return {
            "repr": model.encode(X, **kwargs)
        }

@TEST_METHODS.register("ridge")
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
        # X[mask] = 0
        kwargs["padding_masks"] = mask
        return {
            "repr": model.encode(X, **kwargs),
            "target": target,
        }
