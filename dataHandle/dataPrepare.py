import json

### 按需加载数据
def prepare(ab_flag):
    if ab_flag == 'A':
        variants = [
            'llA',
            'slA',
            'ssA',
            # 'ssB',
            # 'slB',
            # 'llB',
        ]
    else:
        variants = [
            # 'ssA',
            'ssB',
            # 'slA',
            'slB',
            # 'llA',
            'llB',
        ]


    # maxlen = 300

    train_data,valid_data,test_data=[],[],[]
    for i,var in enumerate(variants):
        # i = 0
        key = 'labelA' if 'A' in var else 'labelB'
        if key == 'labelA':
            i = 0
        else:
            i = 1
        #  fs = [
        #     'data/round1/%s/train.txt'%var,
        #     'data/round2/%s.txt'%var,
        #     'data/round3/divided_20210419/%s/train.txt'%var,
        #     'data/rematch/%s/train.txt'%var,
        # ]
        fs = [
            'dataB/round1/%s/train.txt'%var,
            'dataB/round2/%s.txt'%var,
            'dataB/round3/divided_20210419/%s/train.txt'%var,
            'dataB/rematch/%s/train.txt'%var,
        ]
        for f in fs:
            with open(f,encoding='utf-8') as f:
                for l in f:
                    l = json.loads(l)
                    train_data.append((i,l['source'],l['target'],int(l[key])))
        f = 'dataB/rematch/%s/valid.txt'%var
        with open(f,encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                valid_data.append((i,l['source'],l['target'],int(l[key])))
        f = 'dataB/rematch_test_with_id/%s/test_with_id.txt'%var
        with open(f,encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                test_data.append((i,l['source'],l['target'],l['id']))
    print('训练集长度：',len(train_data))
    print('验证集长度：',len(valid_data))
    print('测试集长度：',len(test_data))
    return train_data,valid_data,test_data