import pandas as pd
import os
import datetime
from gensim.models.coherencemodel import CoherenceModel

from gensim import corpora
from gensim import models
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
import argparse
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis


def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())

def main(args):
    ori_data = pd.read_csv(args.data_path)

    # 데이터 전처리
    ori_data_hyun = ori_data[ori_data['3'] == args.brand]  # 현대자동차들
    ori_data_hyun.reset_index(drop=True, inplace=True)

    hyundai = ori_data_hyun

    if args.vehicle == True:
        hyundai_sonata = hyundai[hyundai['4'] == args.vehicle]  # SONATA, SANTA FE
    else:
        hyundai_sonata = hyundai

    hyundai_sonata = hyundai_sonata[hyundai_sonata['7'] >= 20100101]  # 2010년 이전의 데이터는 이상치
    hyundai_sonata = hyundai_sonata.sort_values(by='7', ascending=False)  # 날짜기준 정렬
    hyundai_sonata.reset_index(drop=True, inplace=True)
    hyundai_sonata['19'] = hyundai_sonata['19'].str.lower()  # 대문자 맞추기

    a = hyundai_sonata['19']
    b = hyundai_sonata['7']

    hyundai_sonata = pd.concat([a, b], axis=1)
    hyundai_sonata.rename(columns={'19': 'complaint', '7': 'date'}, inplace=True)

    hyundai_sonata['date'] = hyundai_sonata['date'].astype(str)
    hyundai_sonata['date'] = pd.to_datetime(hyundai_sonata['date'])

    stoplist = [',', '\n', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
                'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    columns = ['cohe', 'perp', 'count']
    Time_data = pd.DataFrame(columns=columns)

    start_day = datetime.datetime.strptime("20100101", "%Y%m%d")

    day1 = datetime.datetime.strptime(args.start_date, "%Y%m%d")  # 시작이 최근
    day2 = datetime.datetime.strptime(args.end_date, "%Y%m%d")

    for i in range(int((day2 - start_day).days), int((day1-start_day).days)):
        start_date  = datetime.datetime(2010, 1, 1) + datetime.timedelta(days=(args.step)* i)
        end_date = start_date + datetime.timedelta(days=args.window)  # 7, 14

        Block = hyundai_sonata[hyundai_sonata['date'] >= start_date]
        Block = Block[Block['date'] < end_date]

        Block_txt = list(Block['complaint'].values)
        texts = [[word for word in document.split() if word.lower() not in stoplist] for document in Block_txt]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=1)

        cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()

        Time_data.loc[len(Time_data)] = [coherence, lda.log_perplexity(corpus), len(Block)]

    Time_shift = Time_data.shift(1).fillna(0)

    Time_data['cohe_diff'] = abs(Time_data['cohe'] - Time_shift['cohe'])
    Time_data['perp_diff'] = abs(Time_data['perp'] - Time_shift['perp'])

    Train_data = Time_data[1:]

    minmax_data = minmax_norm(Train_data)

    minmax_data.to_csv(args.save_path + \
                       f"/{args.end_date}_{args.start_date}_Minmax_Train_data.csv")

    print("학습 Data 생성", args.save_path + \
                       f"/{args.end_date}_{args.start_date}_Minmax_Train_data.csv")
    # Time_data.to_csv(args.save_path + "/Train_data.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=os.path.abspath('./Data/10_22_Data.csv'), help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default=os.path.abspath('./save'), help="model save dir path")
    parser.add_argument(
        "--vehicle", type=str, default=None, help="model save dir path")
    parser.add_argument(
        "--brand", type=str, default='HYUNDAI', help="model save dir path")
    parser.add_argument(
        "--step", type=int, default=1, help="model save dir path")
    parser.add_argument(
        "--window", type=int, default=7, help="model save dir path")

    # 날짜 계산
    parser.add_argument(
        "--start_date", type=str, default='20220401', help="model save dir path")
    parser.add_argument(
        "--end_date", type=str, default='20220301', help="model save dir path")

    args = parser.parse_args()
    main(args)