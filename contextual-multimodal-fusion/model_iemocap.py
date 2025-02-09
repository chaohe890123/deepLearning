import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, \
    multiply, concatenate, MaxPool2D, MaxPooling3D, MaxPooling1D
# sklearn.metrics里的包
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


import numpy as np
import pickle
import sys

def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test

def get_iemocap_raw(classes):
    if sys.version_info[0] == 2:
        f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
    else:
        f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

    # print(len(trainVid))
    # print(len(testVid))

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))
    print('max_len', max_len)
    for vid in trainVid:
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    # print(train_text.shape)
    # print(train_audio.shape)
    # print(train_visual.shape)

    # print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    # print(test_text.shape)
    # print(test_audio.shape)
    # print(test_visual.shape)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(train_seq_len.shape)
    # print(test_seq_len.shape)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask

def get_raw_data(data, classes):
    if data == 'iemocap':
        return get_iemocap_raw(classes)

# 计算测试准确率等结果
def calc_test_result(result, test_label, test_mask, print_detailed_results=True):  # 这里改改，改成打印细节
    '''
    # Arguments
        predicted test labels, gold test labels and test mask

    # Returns
        accuracy of the predicted labels
    '''
    true_label = []
    predicted_label = []

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i, j] == 1:
                true_label.append(np.argmax(test_label[i, j]))
                predicted_label.append(np.argmax(result[i, j]))

    if print_detailed_results:
        print("Confusion Matrix 混淆矩阵:")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report 分类报告 :")
        print(classification_report(true_label, predicted_label))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)

# 将标签转为one-hot形式
def create_one_hot_labels(train_label, test_label):
    '''
    # Arguments
        train and test labels (2D matrices)

    # Returns
        one hot encoded train and test labels (3D matrices)
    '''

    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test

# mask
def create_mask(train_data, test_data, train_length, test_length):
    '''
    # Arguments
        train, test data (any one modality (text, audio or video)), utterance lengths in train, test videos

    # Returns
        mask for train and test data
    '''

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    return train_mask, test_mask


train_data, test_data, train_audio, test_audio, train_text, test_text, train_video, test_video, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(
            "iemocap", 6)

print("-----------------------")
print(train_data.shape)   # (120, 110, 712)
print(train_label.shape)  # (120, 110, 6)
print(test_data.shape)    # (31, 110, 712)
print(test_label.shape)   # (31, 110, 6)




# 跨模态注意力
def bi_modal_attention(x, y):  # x,y分别是两种不同的模态

    '''
    .  stands for dot product
    *  stands for elemwise multiplication
    {} stands for concatenation

    m1 = x . transpose(y) ||  m2 = y . transpose(x)
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y

    return {a1, a2}

    '''

    m1 = dot([x, y], axes=[2, 2])
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    a1 = multiply([o1, x])

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    return concatenate([a1, a2])


def tri_model_attention(v, t, a):     # dense_video, dense_text, dense_audio

    Ftv = Dense(100, activation='tanh')(concatenate([t, v],  axis=2))
    Fta = Dense(100, activation='tanh')(concatenate([t, a], axis=2))
    Fav = Dense(100, activation='tanh')(concatenate([a, v], axis=2))

    c1 = dot([a, Ftv], axes=[2, 2])
    c2 = dot([v, Fta], axes=[2, 2])
    c3 = dot([t, Fav], axes=[2, 2])

    p1 = Activation('softmax')(c1)
    p2 = Activation('softmax')(c2)
    p3 = Activation('softmax')(c3)

    t1 = dot([p1, a], axes=[1, 1])
    t2 = dot([p2, v], axes=[1, 1])
    t3 = dot([p3, t], axes=[1, 1])

    Oatv = multiply([t1, Ftv])
    Ovta = multiply([t2, Fta])
    Otav = multiply([t3, Fav])

    return concatenate([Oatv, Oatv, Oatv], axis=2)

# 自注意力机制
def self_attention(x):
    '''
    .  stands for dot product
    *  stands for elemwise multiplication

    m = x . transpose(x)
    n = softmax(m)
    o = n . x
    a = o * x

    return a

    '''

    m = dot([x, x], axes=[2, 2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2, 1])
    a = multiply([o, x])

    return a


# 上下文注意力模型
def contextual_attention_model(mode):  # 参数分别代表了不同的模型
    ########### Input Layer ############
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))
    ########### Masking Layer ############
    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)
    ########### Recurrent Layer ############
    drop_rnn = 0.7
    gru_units = 300
    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)
    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)
    ########### Time-Distributed Dense Layer ############
    drop_dense = 0.7
    dense_units = 100
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))
    ########### Attention Layer ############
    ## Multi Modal Multi Utterance Bi-Modal attention ##
    if mode == 'MMMU_BA':
        vt_att = bi_modal_attention(dense_video, dense_text)
        av_att = bi_modal_attention(dense_audio, dense_video)
        ta_att = bi_modal_attention(dense_text, dense_audio)
        # merged = concatenate([vt_att, av_att, ta_att, dense_video, dense_audio, dense_text])
        merged = concatenate([ta_att, dense_video, dense_audio, dense_text])
    ## Multi Modal Uni Utterance Self Attention ##
    elif mode == 'MMUU_SA':
        attention_features = []
        for k in range(seqlen_train):
            # extract multi modal features for each utterance #
            m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)
            m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)
            m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)
            utterance_features = concatenate([m1, m2, m3], axis=1)
            attention_features.append(self_attention(utterance_features))
        merged_attention = concatenate(attention_features, axis=1)
        merged_attention = Lambda(lambda x: K.reshape(x, (-1, seqlen_train, 3 * dense_units)))(merged_attention)
        merged = concatenate([merged_attention, dense_video, dense_audio, dense_text])
    ## Multi Utterance Self Attention ##
    elif mode == 'MU_SA':
        vv_att = self_attention(dense_video)
        tt_att = self_attention(dense_text)
        aa_att = self_attention(dense_audio)
        merged = concatenate([aa_att, vv_att, tt_att, dense_video, dense_audio, dense_text])
    ## No Attention ##
    elif mode == 'None':
        merged = concatenate([dense_video, dense_audio, dense_text])
    else:
        print("Mode must be one of 'MMMU-BA', 'MMUU-SA', 'MU-SA' or 'None'.")
        return
    ########### Output Layer ############
    output = TimeDistributed(Dense(6, activation='softmax'))(merged)  # 2分类输出
    print("--------output-------------")
    print(output.shape)
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态
    return model


# TODO:修改的上下文注意力模型
def update_contextual_attention_model(mode):  # 参数分别代表了不同的模型

    ########### Input Layer ############

    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    ########### Masking Layer ############

    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)

    ########### Recurrent Layer ############

    drop_rnn = 0.7
    gru_units = 300

    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    # 可以把视觉特征经过自己的图像分类提取网络，来进行图像特征的提取
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)


    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)


    drop_dense = 0.7
    dense_units = 100

    # 尝试不用这个
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        text_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_text_attention = concatenate(text_attention_features, axis=1)

    # TODO:方案一 只考虑图像模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    video_attention_features = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        video_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_video_attention = concatenate(video_attention_features, axis=1)

    # TODO:方案一  只考虑语音模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    audio_attention_features = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        audio_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_audio_attention = concatenate(audio_attention_features, axis=1)

    # TODO:方案二  跨模态语句级注意力 每个语句的自注意力机制  语句级别的融合  对应相应的重要程度 各模态之间的交互信息
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features1 = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        # TODO:对比试验
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    # 将列表的注意力机制结合起来
    merged_attention1 = concatenate(attention_features1, axis=1)
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, 110, 3 * dense_units)))(
        merged_attention1)

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # TODO:对比实验
    vt_at_concat1 = self_attention(av_att)

    # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    vt_at_concat = concatenate([vt_att, ta_att])
    vt_at_concat_self = self_attention(vt_at_concat)
    vt_self_att = self_attention(vt_att)
    at_self_att = self_attention(ta_att)

    # TODO 方案五 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # TODO:方案三  考虑文本video组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_video = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_tv = concatenate([m1, m3], axis=1)
        attention_features_text_to_video.append(self_attention(utterance_features_tv))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine = concatenate(attention_features_text_to_video, axis=1)
    # reshape
    merged_attention_two_combine = Lambda(lambda x: K.reshape(x, (-1, 110, 2 * dense_units)))(
        merged_attention_two_combine)

    # TODO:方案五  考虑文本audio组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_audio = []
    for k in range(110):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_ta = concatenate([m2, m3], axis=1)
        attention_features_text_to_audio.append(self_attention(utterance_features_ta))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine_ta = concatenate(attention_features_text_to_audio, axis=1)
    # reshape
    merged_attention_two_combine_ta = Lambda(lambda x: K.reshape(x, (-1, 110, 2 * dense_units)))(
        merged_attention_two_combine_ta)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)

    # TODO 方案8 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_audio_attention, merged_video_attention, merged_text_attention,
        merged_attention1,
        av_att, vt_att, ta_att,
        a_v_t_atten,
    ])
    # merged_attention,
    # dense_audio, dense_video, dense_text, merged_attention1
    # ,  vv_att, tt_att, aa_att, a_v_t_atten
    # vt_att, av_att, ta_att

    ########### Attention Layer ############

    ## Multi Modal Multi Utterance Bi-Modal attention ##

    # TODO：跨模态注意力机制  --两两结合
    # print("\n经过两两跨模态注意力之前的shape")
    # print(dense_video.shape)  # (?, 63, 100)
    # print(dense_audio.shape)  # (?, 63, 100)
    # print(dense_text.shape)  # (?, 63, 100)

    # vt_att = bi_modal_attention(dense_video, dense_text)
    # av_att = bi_modal_attention(dense_audio, dense_video)
    # ta_att = bi_modal_attention(dense_text, dense_audio)

    # merged = concatenate([vt_att, av_att, ta_att, dense_video, dense_audio, dense_text])

    # print("\n经过两两跨模态注意力之后的shape")
    # print(vt_att.shape)   # (?, 63, 200)
    # print(av_att.shape)   # (?, 63, 200)
    # print(ta_att.shape)   # (?, 63, 200)

    # TODO：跨模态注意力机制  --三个模态结合   每个模态的重要性不同，直接学习一个矩阵表明重要性程度
    # a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)
    # print("跨模态注意力机制  --三个模态结合之后的shape")
    # print(a_v_t_atten.shape)   # (?, 63, 300)

    # merged = concatenate([vt_att, av_att, ta_att,  a_v_t_atten])
    # print("\n merge shape")
    # print(merged.shape)  # (?, 63, 1200)

    # TODO:经过自注意力机制   每个模态内部重要程度也不同
    # vv_att = self_attention(dense_video)
    # tt_att = self_attention(dense_text)
    # aa_att = self_attention(dense_audio)
    # merged = concatenate([vt_att, av_att, ta_att, a_v_t_atten])

    # # ## Multi Modal Uni Utterance Self Attention ##
    # # TODO：单一语句的自注意力机制   每一个语句的词汇重要性也不同
    # attention_features = []
    #
    # for k in range(max_utt_len):
    #     # extract multi modal features for each utterance #
    #     m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)
    #     m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)
    #     m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)
    #
    #     utterance_features = concatenate([m1, m2, m3], axis=1)
    #     attention_features.append(self_attention(utterance_features))
    #
    # merged_attention = concatenate(attention_features, axis=1)
    # merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention)
    # #     merged = concatenate([merged_attention, dense_video, dense_audio, dense_text])
    #
    # merged = concatenate(
    #     [vt_att, av_att, ta_att, dense_video, dense_audio, dense_text, a_v_t_atten, vv_att, tt_att, aa_att, merged_attention])
    # print("\n merge shape")
    # print(merged.shape)  # (?, 63, 1500)

    ########### Output Layer ############
    # merged = TimeDistributed(Dense(64, activation='softmax'))(merged)   # 80..
    # merged = self_attention(merged)
    output = TimeDistributed(Dense(6, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model


def update_contextual_attention_model2(mode):  # 参数分别代表了不同的模型
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))
    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)
    drop_rnn = 0.7
    gru_units = 300
    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    # 可以把视觉特征经过自己的图像分类提取网络，来进行图像特征的提取
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)
    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)
    drop_dense = 0.7
    dense_units = 100
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        text_attention_features.append(self_attention(m1))
    merged_text_attention = concatenate(text_attention_features, axis=1)

    # 视频每个语句的自注意力机制   获取每个语句内部单词的重要程度
    video_attention_features = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        video_attention_features.append(self_attention(m1))
    merged_video_attention = concatenate(video_attention_features, axis=1)

    # 音频每个语句的自注意力机制   获取每个语句内部单词的重要程度
    audio_attention_features = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        audio_attention_features.append(self_attention(m1))
    merged_audio_attention = concatenate(audio_attention_features, axis=1)

    # TODO:方案二  跨模态语句级注意力 每个语句的自注意力机制  语句级别的融合  对应相应的重要程度 各模态之间的交互信息
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features1 = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    merged_attention1 = concatenate(attention_features1, axis=1)
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(
        merged_attention1)

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # # TODO:对比实验
    # vt_at_concat1 = self_attention(av_att)
    # # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    # vt_at_concat = concatenate([vt_att, ta_att])
    # vt_at_concat_self = self_attention(vt_at_concat)
    # vt_self_att = self_attention(vt_att)
    # at_self_att = self_attention(ta_att)

    # TODO 方案五 三模态的跨模态注意力机制  (?, 126, 300)
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)


    # 维度太高   可以考虑最大池化来降维


    unit = 64
    Bi = concatenate([vt_att, ta_att, av_att], axis=2)
    Ci = Dense(unit, activation="tanh")(Bi)
    Ci = Dense(unit)(Ci)
    a = Activation("softmax")(Ci)
    Bi = multiply([a, Ci])

    Di = concatenate([merged_text_attention, merged_video_attention, merged_audio_attention], axis=2)
    Ti = Dense(unit, activation="tanh")(Di)
    Ti = Dense(unit)(Ti)
    b = Activation("softmax")(Ti)
    Di = multiply([b, Ti])

    Ai = Dense(unit, activation="tanh")(a_v_t_atten)
    Ei = Dense(unit)(Ai)
    c = Activation("softmax")(Ei)
    Ai = multiply([c, Ei])

    # dense_audio = multiply([dense_audio, merged_audio_attention])
    # dense_video = multiply([dense_video, merged_video_attention])
    # dense_text = multiply([dense_text, merged_text_attention])

    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_text_attention, merged_video_attention, merged_audio_attention,
        # Di,
        # vt_att, ta_att,  # av_att,
        Bi ,
        tt_att,
        # Ai,
        a_v_t_atten,
    ], axis=2)

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model



# 训练模型
def train(mode):
    runs = 5  # 求五次结果平均
    accuracy = []

    for j in range(runs):
        # compile model #
        # 原始论文模型
        # model = contextual_attention_model(mode)
        # 第一次小论文修改模型
        model = update_contextual_attention_model(mode)
        # 第二次小论文修改模型
        # model = update_contextual_attention_model2(mode)
        # 打印出模型summary
        # model.summary()
        LEARNING_RATE = 0.001

        optimizer = Adam(lr=LEARNING_RATE)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal',
                      metrics=['accuracy'])

        # set callbacks #
        path = 'weights/Mosi_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'  # 报错，需要建立weights目录

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=10, verbose=1, min_lr=0.00001)
        # early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0)  # 将val_loss 改成了 val_acc
        check = ModelCheckpoint(path, monitor='val_acc', save_best_only=True, mode='max', verbose=0)

        # train model #
        history = model.fit([train_text, train_audio, train_video], train_label,
                            epochs=128,  # 感觉可以加大epoch
                            batch_size=32,  # 改成64好像效果不是很好
                            sample_weight=train_mask,
                            shuffle=False,   # True
                            callbacks=[check, reduce_lr],   # early_stop
                            # validation_data=([dev_text, dev_audio, dev_video], dev_label, dev_mask),
                            validation_data=([test_text, test_audio, test_video], test_label, test_mask),
                            verbose=1)

        # test results #
        model.load_weights(path)
        test_predictions = model.predict([test_text, test_audio, test_video])
        test_accuracy = calc_test_result(test_predictions, test_label, test_mask)
        accuracy.append(test_accuracy)

        # release gpu memory #
        K.clear_session()
        del model, history
        gc.collect()

    # summarize test results #

    avg_accuracy = sum(accuracy) / len(accuracy)
    max_accuracy = max(accuracy)

    print('Mode: ', mode)
    print('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:', '{0:.4f}'.format(max_accuracy))
    print('-' * 55)


if __name__ == "__main__":

    # for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:
    #     train(mode)
    mode = "sdq"
    train(mode)