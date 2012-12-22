#!/usr/bin/env python
# coding: utf-8
"""
Multinomial Naive Bayes
"""
import sys
import json
from math import log
from optparse import OptionParser

def prob(file):
    """
    LIBSVM形式のファイルを読み込む
    """
    y, x = [], []
    for line in open(file):
        tokens = line.rstrip().split(' ')
        cls = int(tokens[0])
        vec = {}
        for token in tokens[1:]:
            item = token.split(':')
            key = int(item[0])
            val = float(item[1])
            vec[key] = val
        y.append(cls)
        x.append(vec)
    return y, x

def train(y, x, alpha=1.0):
    """
    学習
    """
    N_c = {}  # クラスごとの文書数
    n_c = {}  # クラスごとの単語数
    n_wc = {} # 単語wのクラスcにおける出現数

    for i, c in enumerate(y):
        # クラスcの文書数をカウント
        N_c[c] = N_c.get(c, 0) + 1
        for w, num in x[i].items():
            # クラスcの単語数をカウント
            n_c[c] = n_c.get(c, 0.0) + num
            # 単語wのカウント
            if w in n_wc:
                n_wc[w][c] = n_wc[w].get(c, 0.0) + num
            else:
                n_wc[w] = { c: num }

    N = len(y)           # 全文書数
    C = len(N_c.keys())  # 全クラス数
    W = len(n_wc.keys()) # 全単語種数

    # モデル構築
    p_c = {} # クラス事前確率
    for c, v in N_c.items():
        p_c[c] = log(v + alpha) - log(N + C * alpha)
    q_wc = {} # 単語の独立確率分布
    for w in n_wc.keys():
        q_wc[w] = {}
        for c in N_c.keys():
            q_wc[w][c] = log(n_wc[w].get(c, 0.0) + alpha) - log(n_c.get(c, 0.0) + W * alpha)

    # モデルを返す
    return { 'p_c': p_c, 'q_wc': q_wc }

def predict(y, x, model):
    """
    予測
    """
    tnum = 0    # 正解数
    p_labs = [] # 推定ラベル
    p_vals = [] # 対数尤度
    p_c  = model['p_c']
    q_wc = model['q_wc']

    # 推定
    for i, tc in enumerate(y):
        p_val = []
        max = -1 * sys.maxint
        for c, v in sorted(p_c.items()):
            for w, num in x[i].items():
                if w in q_wc:
                    # 単語が存在すればクラス事前確率に足し込み
                    v += num * q_wc[w][c]
            if v > max:
                max = v
                max_c = c
            p_val.append(v)
        # 推定クラスを記録
        p_labs.append(max_c)
        # 対数尤度を記録
        p_vals.append(p_val)
        # 正解チェック
        if tc == max_c:
            tnum += 1

    # 正解率の計算
    acc = 100.0 * tnum / len(y)

    # 推定クラス、正解率、対数尤度を返す
    return p_labs, acc, p_vals

def load(file):
    """
    モデルを読み込む
    """
    return json.load(open(file), object_hook=_decode)

def save(model, file):
    """
    モデルを書き出す
    """
    json.dump(model, open(file, 'w'))

def _decode(data):
    """
    int:floatのdictionaryを作るためのデコーダ
    """
    d = {}
    for k, v in data.items():
        # p_c, q_wc以外のキーはintとして読み込む
        if k != u'p_c' and k != u'q_wc':
            k = int(k)
        d[k] = v
    return d

# メイン関数
def main():
    usage = 'usage: %prog --train or --predict [options] data model'
    parser = OptionParser(usage=usage)
    parser.add_option('--train', dest='train', help='training mode', action='store_true', default=False)
    parser.add_option('--predict', dest='predict', help='predict mode', action='store_true', default=False)
    parser.add_option('-a', dest='alpha', help='alpha', type='float', default=1.0)
    options, args = parser.parse_args()

    # 入力チェック
    if len(args) != 2:
        parser.error('incorrect number of arguments')
    if not (options.train or options.predict):
        parser.error('--train or --predict is need to execute')

    # trainモード
    if options.train:
        y, x = prob(args[0])
        model = train(y, x, options.alpha)
        save(model, args[1])
    # predictモード
    elif options.predict:
        y, x = prob(args[0])
        model = load(args[1])
        p_labs, p_acc, p_vals = predict(y, x, model)
        print 'Accuracy %8.4f%%' % (p_acc)

# エントリポイント
if __name__ == '__main__':
    main()

