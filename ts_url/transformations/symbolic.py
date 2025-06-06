from ..registry import TRANSFORMATION
from sktime.transformations.panel.dictionary_based import SFA
from transformers import AutoModel
from transformers import RobertaTokenizer, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
# tokenizer = RobertaTokenizer.from_pretrained("allenai/longformer-base-4096")
from .ts_transformation import VoidTransfromation
import os 
import torch
from pyts.transformation import WEASEL
import numpy as np
from torch import nn

__all__ = ["SymbolicFA", "WEASELbolicFA"]

def zero_except_topk(matrix, k):
    """
    将矩阵中每一行除了topk的元素都置为0

    参数:
    matrix (np.ndarray): 输入的numpy矩阵
    k (int): 保留的topk元素的数量

    返回:
    np.ndarray: 处理后的矩阵
    """
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)
    for i in range(rows):
        row = matrix[i]
        topk_indices = np.argpartition(row, -k)[-k:]
        result[i, topk_indices] = row[topk_indices]

    return result

def convert_to_ordinal(num):
    # 特殊情况处理 11, 12, 13
    if 10 <= num % 100 <= 13:
        suffix = 'th'
    else:
        # 根据数字的最后一位决定后缀
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
    return str(num) + suffix

def get_weasel_rep(Xtrain, y_train, tokenizer, maXchannel=20, maXbag=5):
    channel_word_bags = []
    start_prompt = "Symbolic fourier transformed words presented as lists of (word frequency, window size, word). "
    # print(Xtrain.shape)
    for d in range(min(Xtrain.shape[1], maXchannel)):
        X = Xtrain[:, d]
        
        weasel = WEASEL(word_size=2, window_sizes=[0.3, 0.5, 0.7], n_bins=2, sparse=False)
        Xweasel = weasel.fit_transform(X, y_train)
        Xweasel = zero_except_topk(Xweasel, maXbag)

        # Visualize the transformation for the first time series
        words = np.vectorize(weasel.vocabulary_.get)(np.arange(Xweasel[0].size))
        n_samples, alphabet_size = Xweasel.shape
        
        bags_persample = []
        for sample_id in range(n_samples):
            sample_words = []
            for alpha in range(alphabet_size):
                if Xweasel[sample_id, alpha] != 0:
                    sample_words.append("(" + ", ".join((str(Xweasel[sample_id, alpha]), ) + tuple(words[alpha].split(" ", 1))) + ")")
            bags_persample.append(sample_words)
        
        channel_word_bags.append(bags_persample)
    # print(len(channel_word_bags[0]))
    # print(len(channel_word_bags))

    channel_transform = lambda c,bags:f"Words for the {convert_to_ordinal(c)} channel: {bags} " 
    transformed = []
    tokens = []
    for sample_id in range(Xtrain.shape[0]):
        trans = start_prompt
        for c in range(min(Xtrain.shape[1], maXchannel)):
            
            # Visualize the transformation for the first time series
            trans += channel_transform(c, "[" + ", ".join(channel_word_bags[c][sample_id]) + "]")
        transformed.append(trans)
        # print(trans)
        token_ids = tokenizer(trans, return_tensors="pt")['input_ids']
        
        tokens.append(token_ids[0])
        # print(tokens[-1].shape)
    return tokens


@TRANSFORMATION.register("sfa")
class SymbolicFA(VoidTransfromation):
    def __init__(self, X, d_name, maXchannel=20, dset_type="train", tokenizer="allenai/longformer-base-4096",**kwargs):

        def list2string(word):
            string_ = ""
            for w in word:
                string_ += str(w)
            return string_

        sfa = SFA()
        sfas = []
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        space = tokenizer.tokenize(" ")[0]
        save_path = f"augmentation/sfa_{d_name}_{maXchannel}_{dset_type}.pt"
        if os.path.exists(save_path):
            self.sfas = torch.load(save_path)
            return None
        else:
            for b in range(X.shape[0]):
                tokens_all = []
                for d in range(min(X.shape[1], maXchannel)):
                    results = sfa.fit_transform(X[b][d][None, None])[0][0]
                    prompt_info = []
                    for word, num in results.items():
                        prompt_info.append(
                            (sfa.word_list_typed(word), num)
                        )
                    prompt = f"The most frequent words in the bag for the {d}th channel are: " 
                    prompt_info = sorted(prompt_info, key=lambda x:x[1], reverse=True)

                    words_tokens = []
                    for word, num in prompt_info[:min(5, len(prompt_info))]:
                        word = list2string(word)
                        words_tokens.append(",")
                        words_tokens.append(space)
                        words_tokens.append(str(num))
                        words_tokens.append(space)
                        words_tokens += word
                    words_tokens.append('.')
                    words_tokens = words_tokens[2:]

                    tokens_ = tokenizer.tokenize(prompt) + words_tokens
                    tokens_all.append(space)
                    tokens_all += tokens_
                sfas.append(tokens_all)

            tokenized = []
            for sf in sfas:
                sf = tokenizer.encode(sf, return_tensors="pt")
                tokenized.append(sf[0])
            sfas = pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.encode("<PAD>")[0])
            assert not torch.isnan(sfas).any().item()
            torch.save(sfas, save_path)
        self.sfas = sfas
        # print(sfas.shape)
        # exit()
    
    def transform(self, x, index, **kwargs):
        return self.sfas[index].unsqueeze(0)
    
    
@TRANSFORMATION.register("weasel")
class WEASELbolicFA(VoidTransfromation):
    def __init__(self, X, y, d_name, maXchannel=8, maXbag=5, dset_type="train", tokenizer="google/bigbird-roberta-base", **kwargs):

        save_path = f"augmentation/weasel_{d_name}_{maXchannel}_{dset_type}.pt"
        if os.path.exists(save_path):
            self.sfas = torch.load(save_path)
            return None
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            
            if X.shape[1] > maXchannel:
                X = torch.tensor(X)
                adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=(64, X.shape[-1]))
                X = adaptive_avgpool(X)
                X = X.numpy()
                
            tokenized = get_weasel_rep(X, y, maXchannel=maXchannel, maXbag=maXbag, tokenizer=tokenizer)
            sfas = pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.encode("<PAD>")[0])
            torch.save(sfas, save_path)
        self.sfas = sfas
        # print(sfas.shape)
        # exit()
    
    def transform(self, x, index, **kwargs):
        return self.sfas[index].unsqueeze(0)