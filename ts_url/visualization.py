import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import threading

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.animation import FuncAnimation
# 生成一些随机损失数据

def smoothing_(loss, smoothing_rate):
    base_smooth = len(loss) // 3
    smoothing_len = int(max(1, base_smooth * smoothing_rate))
    padding_size = (smoothing_len - 1)// 2
    smoothing = np.ones(smoothing_len) / smoothing_len
    padding_right = [(0, padding_size)]
    padded = np.pad(loss, padding_right, mode="constant", constant_values=loss[-1])
    if smoothing_len % 2 == 0:
        padding_size += 1
    padding_left = [(padding_size, 0)]
    padded = np.pad(padded, padding_left, mode="constant", constant_values=loss[0])
    return np.convolve(padded, smoothing, mode="valid")

def plot_loss(train_loss, val_loss):
    # 绘制初始的损失趋势图
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    train_line, = ax.plot(train_loss, label='Train Loss')
    val_line, = ax.plot(val_loss, label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # # 让用户调节曲线的平滑度
    train_smoothing = 0.2
    val_smoothing = 0.2

    def update_smooth(val):
        # global train_smoothing, val_smoothing
        nonlocal train_smoothing, val_smoothing
        train_smoothing, val_smoothing = val
        smoothed_train_loss = smoothing_(train_loss, train_smoothing)
        smoothed_val_loss = smoothing_(val_loss, val_smoothing)
        train_line.set_ydata(smoothed_train_loss)
        val_line.set_ydata(smoothed_val_loss)
        fig.canvas.draw_idle()

    # 创建两个滑块，用于调整训练和验证损失的平滑度
    ax_train_smooth = plt.axes([0.2, 0.10, 0.65, 0.03])
    slider_train_smooth = plt.Slider(ax_train_smooth, 'Train Smoothing', 0.1, 1.0, valinit=train_smoothing)

    ax_val_smooth = plt.axes([0.2, 0.07, 0.65, 0.03])
    slider_val_smooth = plt.Slider(ax_val_smooth, 'Val Smoothing', 0.1, 1.0, valinit=val_smoothing)

    slider_train_smooth.on_changed(lambda val: update_smooth([val, val_smoothing]))
    slider_val_smooth.on_changed(lambda val: update_smooth([train_smoothing, val]))

    # 绘制平滑后的训练和验证损失趋势图
    smoothed_train_loss = smoothing_(train_loss, train_smoothing)
    smoothed_val_loss = smoothing_(val_loss, val_smoothing)
    train_line.set_ydata(smoothed_train_loss)
    val_line.set_ydata(smoothed_val_loss)

    plt.show()
# num_epochs = 50
# train_loss = np.random.rand(num_epochs)
# val_loss = np.random.rand(num_epochs)
# plot_loss(num_epochs, train_loss, val_loss)


def visualize_kmeans_clusters(kmeans_labels, original_matrix):
    # 使用t-sne进行降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    embedded_matrix = tsne.fit_transform(original_matrix)

    # 绘制散点图
    plt.scatter(embedded_matrix[:, 0], embedded_matrix[:, 1], c=kmeans_labels)
    plt.show()

# embedding_matrix = np.random.rand(100, 50)

# # 使用KMeans进行聚类
# kmeans = KMeans(n_clusters=3, random_state=0)
# kmeans.fit(embedding_matrix)
# labels = kmeans.labels_

# # 可视化聚类结果
# visualize_kmeans_clusters(labels, embedding_matrix)


def visualize_prediction(input_seq, pred_seq):
    """
    可视化一个时间序列的输入序列和模型预测序列。
    参数：
    - input_seq：一个numpy数组，形状为(n, d)，表示d维时间序列的输入序列。
    - pred_seq：一个numpy数组，形状为(k, d)，表示d维时间序列的模型预测序列。
    """
    n, d = input_seq.shape
    k = pred_seq.shape[0]
    orig_d = d
    # 仅展示前20个维度
    if d > 10:
        d = 10
        input_seq = input_seq[:, :d]
        pred_seq = pred_seq[:, :d]
        
    # 设置子图的大小
    fig, axs = plt.subplots(d, 1, figsize=(8, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 对每个维度的时间序列进行可视化
    for i in range(d):
        # 获取第i维的时间序列
        input_ts = input_seq[:, i]
        pred_ts = pred_seq[:, i]
        # 绘制输入序列和预测序列
        axs[i].plot(range(n-k + 1), input_ts[:n-k + 1], color='blue', label='Input')
        axs[i].plot(range(n-k, n), input_ts[n-k:], color='green', label='Ground Truth')
        axs[i].plot(range(n-k, n), pred_ts, color='red', label='Prediction')
        # 添加图例和标签
        axs[i].legend(fontsize=8)
        axs[i].set_ylabel('Dim {}'.format(i+1))
        
    # 添加共享的x轴标签
    plt.xlabel('Time')
    
    # 如果展示的维度小于总维度，则在右下角添加提示信息
    if d < orig_d:
        ax_text = plt.axes([0.1, 0.9, 0.8, 0.1])
        ax_text.axis('off')
        text = ax_text.text(0.5, 0.5, 'only show {} dimentions'.format(d), ha='center', va='center', transform=ax_text.transAxes)
    # 显示图像
    plt.show()

# # 生成一个随机的输入序列和预测序列作为示例
# n, k, d = 20, 10, 3
# input_seq = np.random.randn(n + k, d)
# pred_seq = np.random.randn(k, d)

# # 可视化时间序列
# visualize_prediction(input_seq, pred_seq)

def plot_imputation(time_series, mask, reconstructed):
    """
    生成维度曲线图和掩码的可视化。
    Args:
        time_series (numpy.ndarray): 原始时间序列，形状为(n, d)。
        mask (numpy.ndarray): 伯努利分布的掩码，形状为(n, d)。
        reconstructed (numpy.ndarray): 重构的时间序列，形状为(n, d)。
    """
    num_dims = time_series.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.8)

    for i in range(num_dims):
        # 绘制原始时间序列
        axes[i].plot(time_series[:, i], label='Original', color='blue')

        # 在原始时间序列上绘制被掩码遮挡的时间步
        masked_indices = np.argwhere(mask[:, i] == 0).flatten()
        masked_values = reconstructed[:, i][masked_indices]
        axes[i].plot(masked_indices, masked_values, 'o', label='Masked Values', color='red')

        # 在原始时间序列上绘制被掩码遮挡的时间步
        masked_indices = np.argwhere(mask[:, i] == 0).flatten()
        masked_values = time_series[:, i][masked_indices]
        axes[i].plot(masked_indices, masked_values, 'o', label='True Values', color='green')

        reconstructed_indices = np.argwhere(mask[:, i] == 0).flatten()
        for idx in reconstructed_indices:
            axes[i].axvline(x=idx, linestyle='--', color='green')

        # 设置子图标题、标签和图例
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Value')
        axes[i].legend()
    plt.xlabel("Time")
    plt.show()


# 示例用法
# time_series = np.random.rand(100, 4)
# mask = np.random.binomial(1, 0.5, size=(100, 4))
# reconstructed = np.random.rand(100, 4)
# plot_imputation(time_series, mask, reconstructed)

def refine_list(label, x, alpha):
    start_idx = None
    end_idx = None
    new_label = []
    new_x = []

    for i in range(len(label)):
        if label[i] == 1:
            # 如果找到了一个新的1区间，则在该区间两侧插入1和0
            if start_idx is None:
                start_idx = i
                end_idx = i
                new_label.extend([0, 1])
                new_x.extend([x[i] - 2*alpha, x[i] - alpha])
            else:
                end_idx = i
            
        else:
            # 如果1区间结束了，则在该区间两侧插入0和1
            if start_idx is not None:
                new_label.extend([1, 0])
                new_x.extend([x[i] + alpha, x[i] + 2*alpha])
                start_idx = None
                end_idx = None
        new_label.append(label[i])
        new_x.append(x[i])

        # 处理最后一个1区间
        if i == len(label) - 1 and start_idx is not None:
            new_label.extend([1, 0])
            new_x.extend([x[i] + alpha, x[i] + 2*alpha])

    return new_label, new_x


label = [0, 1, 1, 0, 1, 0, 0, 1, 1]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
alpha = 0.1
# print(refine_list(label, x, alpha))
def draw_interval(ax, labels, upper, lower,  color='red'):
    fill_x = list(range(len(labels)))
    vis_labels = list(labels)
    vis_labels, fill_x = refine_list(vis_labels, fill_x, 0.01)
    vis_labels = np.array(vis_labels)
    return ax.fill_between(fill_x, lower * vis_labels, upper * vis_labels , alpha=0.3, color=color, label='Anomaly Interval')

def visualize_scores(scores, labels, input_seq):
    # Remove outliers
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    scores = np.clip(scores, None, threshold)

    # Plot the scores and anomaly intervals
    n, d = input_seq.shape
    if d > 10:
        d = 10
    fig, ax = plt.subplots(d + 1, 1, figsize=(12, 8), sharex=True)
    axs = ax[1:]
    ax = ax[0]
    fig.subplots_adjust(bottom=0.25)
    max_score = np.max(scores)
    min_score = np.min(scores)
    ax.plot(scores, label='Score', color='red')
    ax.plot([], label='Time Series', color='blue')
    ax.set_ylabel("Scores")
    draw_interval(ax, labels, 0 , max_score)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 3.6))

    # Add a slider and calculate accuracy, precision, recall, and F1 score
    ax_slider = plt.axes([0.1, 0.15, 0.8, 0.05])
    slider = plt.Slider(ax_slider, 'Threshold', np.min(scores), np.max(scores) * 2 / 3, valinit=np.mean(scores))
    ax_slider_left = plt.axes([0.1, 0.10, 0.8, 0.05])
    ax_slider_right = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider_left = plt.Slider(ax_slider_left, 'left', 0, len(scores), valinit=0)
    slider_right = plt.Slider(ax_slider_right, "right", 0, len(scores), valinit=len(scores))
    # Save all detection result intervals
    detections = []

    # Add a text box to display the performance metrics
    ax_text = plt.axes([0.1, 0.9, 0.8, 0.1])
    ax_text.axis('off')
    text = ax_text.text(0.5, 0.5, '', ha='center', va='center', transform=ax_text.transAxes)
    axs = vis_raw_time_series(input_seq, axs)
    input_seq_max = list(np.max(input_seq, axis=0))
    input_seq_min = list(np.min(input_seq, axis=0))
    for axi, max_, min_ in zip(axs, input_seq_max, input_seq_min):
        draw_interval(axi, labels, max_, min_)

    orig_threshold = 0
    def update(val):
        nonlocal labels
        nonlocal orig_threshold

        threshold = slider.val
        predicted_labels = np.where(scores > threshold, 1, 0)
        accuracy = metrics.accuracy_score(labels, predicted_labels)
        precision = metrics.precision_score(labels, predicted_labels)
        recall = metrics.recall_score(labels, predicted_labels)
        f1_score = metrics.f1_score(labels, predicted_labels)
        text.set_text(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')
        nonlocal detections

        if threshold != orig_threshold:
            # Clear all existing fill_between areas
            for detection in detections:
                detection.remove()
            detections = []

            # Plot new fill_between area
            fill_x = list(range(len(labels)))
            vis_labels = list(predicted_labels)
            vis_labels, fill_x = refine_list(vis_labels, fill_x, 0.01)
            vis_labels = np.array(vis_labels)
            detection = ax.fill_between(fill_x, 0, vis_labels * max_score, alpha=0.3, color='green', label='Detection Result')
            detections.append(detection)
            ax.legend(loc='upper right', bbox_to_anchor=(1, 3.6))
            for axi, max_, min_ in zip(axs, input_seq_max, input_seq_min):
                det = draw_interval(axi, predicted_labels, max_, min_, "green")
                detections.append(det)
            orig_threshold = threshold

        xlim_left = slider_left.val
        xlim_right = slider_right.val
        ax.set_xlim([xlim_left, xlim_right])

        for axi, max_, min_ in zip(axs, input_seq_max, input_seq_min):
            axi.set_xlim([xlim_left, xlim_right])
    
    slider.on_changed(update)
    slider_left.on_changed(update)
    slider_right.on_changed(update)
    # 创建一个FuncAnimation对象，在每一帧中更新第二个窗口中的曲线
    # def update_figure(frame):
    #     fig2.canvas.draw_idle()

    # ani = FuncAnimation(fig2, update_figure, interval=10, blit=False)
    plt.show()

# # 生成示例数据
# n = 100
# scores = np.random.randn(n)
# labels = np.zeros(n)
# labels[20:40] = 1
# labels[60:80] = 1

# # 绘制异常分数和异常区间的图形，并添加滑块
# visualize_scores(scores, labels)


def plot_pie_chart(logits, target):
    target = int(target)
    # Perform softmax to convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Sort the probabilities and find the indices of the smallest ones
    sorted_probs = np.sort(probs)
    sorted_indices = np.argsort(probs)

    # Find the sum of probabilities that are smaller than 0.1
    prob_sum = 0
    for i in range(len(sorted_probs)):
        prob_sum += sorted_probs[i]
        if prob_sum > 0.1:
            sum_up_idx = i 
            break
    all_idx = list(range(len(sorted_probs)))
    small_prob_indices = (all_idx[:sum_up_idx], )
    big_prob_indices = (all_idx[sum_up_idx : ], )
    small_prob_sum = np.sum(sorted_probs[small_prob_indices])

    # Create labels for the pie chart
    labels = [str(i) for i in sorted_indices]
    if small_prob_sum > 0:
        small_prob_labels = [labels[i] for i in small_prob_indices[0]]
        small_prob_label = ','.join(small_prob_labels)

        labels = [label if i not in small_prob_indices[0] else small_prob_label for i, label in enumerate(labels)]
        labels = labels[-1-len(big_prob_indices[0]):]
        label_modified = False
        for i in range(len(labels) - 1):
            print(labels, target)
            if labels[-i] == str(target):
                labels[-i] = str(target) + " (Ground Truth)"
                label_modified = True
        if not label_modified:
            labels[0] = labels[0] + " (Ground Truth)"
            

    # Create a list of colors for the pie chart
    colors = ['C' + str(i) for i in range(len(sorted_indices))]

    # Create a list of sizes for the pie chart
    sizes =  [small_prob_sum] + list(sorted_probs[big_prob_indices])

    # Create the pie chart with percentage labels
    plt.pie(sizes, labels=labels, colors=colors, labeldistance=1.15, autopct='%1.1f%%')
    plt.axis('equal')

def vis_raw_time_series(input_seq, axs_in=None):
    n, d = input_seq.shape
    # 仅展示前10个维度
    orig_d = d
    if d > 10:
        d = 10
        input_seq = input_seq[:, :d]
    if axs_in is None:
        fig, axs = plt.subplots(d, 1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(hspace=0.3)
    else:
        axs = axs_in
    # 对每个维度的时间序列进行可视化
    for i in range(d):
        # 获取第i维的时间序列
        input_ts = input_seq[:, i]
        # 绘制输入序列
        if axs_in is not None:
            axs[i].plot(range(n), input_ts, color='blue')
        else:
            axs[i].plot(range(n), input_ts, color='blue', label="Time Series")
            
        # 添加标签
        axs[i].set_ylabel('Dim {}'.format(i+1))
    if axs_in is None:
        axs[0].legend(loc='upper right')
    # 如果展示的维度小于总维度，则在右下角添加提示信息
    if d < orig_d:
        ax_text = plt.axes([0.1, 0.92, 0.8, 0.1])
        ax_text.axis('off')
        text = ax_text.text(0.5, 0.5, 'only show {} dimentions'.format(d), ha='center', va='center', transform=ax_text.transAxes)
    # 添加共享的x轴标签
    plt.xlabel('Time')
    if axs_in is None:
        return axs, fig
    else:
        return axs

def visualize_classification(input_seq, logits, target):
    """
    可视化一个时间序列的不同维度和分类的logits。
    参数：
    - input_seq：一个numpy数组，形状为(n, d)，表示d维时间序列的输入序列。
    - logits：一个numpy数组，形状为(k,)，表示分类的logits。
    """
    
    k = logits.shape[0]

    plot_pie_chart(logits, target)
    vis_raw_time_series(input_seq)

    # 显示图像
    plt.show()

def visualize_sample_(per_batch, task_name,sample_index):
    if task_name == "regression":
        targets = per_batch["X"][sample_index]
        predictions = per_batch["predictions"][sample_index]
        if len(targets.shape) == 3:
            targets = targets.reshape(targets.shape[-2:])
            predictions = predictions.reshape(predictions.shape[-2:])
        visualize_prediction(targets, predictions)
    # raise NotImplementedError
    elif task_name == "imputation":
        targets = per_batch["targets"][sample_index]
        predictions = per_batch["predictions"][sample_index]
        target_masks = per_batch["target_masks"][sample_index]
        if len(targets.shape) == 3:
            target_masks = target_masks.reshape(target_masks.shape[-2:])
            targets = targets.reshape(targets.shape[-2:])
            predictions = predictions.reshape(predictions.shape[-2:])
        plot_imputation(targets, target_masks, predictions)
    elif task_name == "anomaly_detection":
        targets = per_batch["targets"].reshape(-1)
        scores = per_batch["score"].reshape(-1)
        input_seq = per_batch["X"][:, 0, -1, :]
        if len(input_seq.shape) == 3:
            input_seq = input_seq.reshape(input_seq.shape[-2:])
        visualize_scores(scores, targets, input_seq)
    elif task_name == "clustering":
        reps = per_batch["predictions"]
        if len(reps.shape) == 3:
            reps = reps.reshape(reps.shape[0], reps.shape[2])
        rst = per_batch["clustering_rst"]
        visualize_kmeans_clusters(rst, reps)
    elif task_name == "classification":
        input_seq = per_batch["X"][sample_index]
        predictions = per_batch["predictions"][sample_index]
        target = per_batch["targets"][sample_index]
        if len(input_seq.shape) == 3:
            input_seq = input_seq.reshape(input_seq.shape[-2:])
        if len(predictions.shape) == 2:
            predictions = predictions.reshape(-1)
        visualize_classification(input_seq, predictions, target)
    else:
        raise NotImplementedError()
    
