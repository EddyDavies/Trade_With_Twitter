import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# def create(path):
#     tokenize = lambda x: x.split()
#
#     tweet = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
#     score = Field(seqential=False, use_vocab=False)
#
#     fields = {'tweet': ('t', tweet), 'score': ('s', score)}
#
#     train_data, test_data = TabularDataset.splits(
#         path=path, train='train.csv', validation='validation.csv', test='test.csv', format='csv', fields=fields)
#
#
# from farm.data_handler.data_silo import DataSilo
# from farm.data_handler.processor import TextClassificationProcessor
# from farm.modeling.optimization import initialize_optimizer
# from farm.infer import Inferencer
# from farm.modeling.adaptive_model import AdaptiveModel
# from farm.modeling.language_model import LanguageModel
# from farm.modeling.prediction_head import MultiLabelTextClassificationHead
# from farm.modeling.tokenization import Tokenizer
# from farm.train import Trainer
# from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
# import logging
#
#
# ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
# ml_logger.init_experiment(experiment_name="BBC_Articles", run_name="BBC News Articles")
#
# set_all_seeds(seed=42)
# device, n_gpu = initialize_device_settings(use_cuda=True)
# n_epochs = 2
# batch_size = 8
# evaluate_every = 100
#
#
#
# lang_model = "vinai/bertweet-base"
# do_lower_case = False
#
# tokenizer = Tokenizer.load(
#     pretrained_model_name_or_path=lang_model,
#     do_lower_case=do_lower_case)
#
# label_list = [0, 2, 4] #labels in our data set
# metric = "f1_macro" # desired metric for evaluation
#
# processor = TextClassificationProcessor(tokenizer=tokenizer,
#                                             max_seq_len=512, # BERT can only handle sequence lengths of up to 512
#                                             data_dir='../data/fine_tune',
#                                             label_list=label_list,
#                                             label_column_name="sentiment", # our labels are located in the "genre" column
#                                             metric=metric,
#                                             quote_char='"',
#                                             multilabel=True,
#                                             train_filename="train.tsv",
#                                             dev_filename=None,
#                                             test_filename="test.tsv",
#                                             dev_split=0.1 # this will extract 10% of the train set to create a dev set
#                                             )
#
#
# data_silo = DataSilo(
#     processor=processor,
#     batch_size=batch_size)
#
#
# # loading the pretrained BERT base cased model
# language_model = LanguageModel.load(lang_model)
# # prediction head for our model that is suited for classifying news article genres
# prediction_head = MultiLabelTextClassificationHead(num_labels=len(label_list))
#
# model = AdaptiveModel(
#         language_model=language_model,
#         prediction_heads=[prediction_head],
#         embeds_dropout_prob=0.1,
#         lm_output_types=["per_sequence"],
#         device=device)
#
# model, optimizer, lr_schedule = initialize_optimizer(
#         model=model,
#         learning_rate=3e-5,
#         device=device,
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs)
#
# trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device)
#
# save_dir = "saved_models/bert-english-news-article"
# model.save(save_dir)
# processor.save(save_dir)
# In [0]:
# # to download the model
# !zip -r saved_models/model.zip saved_models/bert-english-news-article
# In [0]:
# inferenced_model = Inferencer.load(save_dir)
# In [0]:
# def read_file(file_name: str) -> dict:
#   text_file = open (file_name, 'r')
#   text_file = text_file.read().replace('\n', ' ')
#   return {'text': text_file}
# In [0]:
# def create_input(text_files:list) -> list:
#   model_input = list()
#   for text_file in text_files:
#     model_input.append(read_file(text_file['file']))
#   return model_input
# In [0]:
# def create_result_overview (articles:list, result:list) -> pd.DataFrame:
#   files = list()
#   labels = list()
#   predictions = list()
#   for i in range(len(articles)):
#     files.append (articles[i]['file'])
#     labels.append(articles[i]['genre'])
#     predictions.append(result[0]['predictions'][i]['label'].strip("'[]'"))
#   data = {'file': files, 'actual': labels, 'prediction': predictions}
#   df = pd.DataFrame(data)
#   return df
# In [0]:
# articles = [{'file': 'bbc_news/generated_data/inferencing/business.txt', 'genre': 'business'},
#             {'file': 'bbc_news/generated_data/inferencing/sport.txt', 'genre': 'sport'}]
#
# article_texts = create_input(articles)
#
# result = inferenced_model.inference_from_dicts(article_texts)
#
# df = create_result_overview(articles, result)
#
# df.head()
#


def split(raw_path, fine_tune_path, sample_percent=None):

    DATASET_ENCODING = "ISO-8859-1"
    print("loading...")
    df = pd.read_csv(raw_path, encoding=DATASET_ENCODING)
    df = df.iloc[:, [0, 5]]

    cols = list(df.columns)
    cols[1], cols[0] = cols[0], cols[1]
    df = df[cols]

    if sample_percent is None:
        train, test = train_test_split(df, test_size=0.2, random_state=1)
    else:
        sample = int(float(sample_percent) * int(df.shape[0]))
        train, test = train_test_split(df.iloc[:sample, :], test_size=0.2, random_state=1)
        
    header = ["tweet", "sentiment"]

    if not os.path.exists(fine_tune_path):
        os.makedirs(fine_tune_path)

    train.to_csv(fine_tune_path + "/train.csv", index=False, header=header, encoding='utf-8')
    test.to_csv(fine_tune_path + "/test.csv", index=False, header=header, encoding='utf-8')
    print(train.shape[0], test.shape[0])


def remove_long(raw_path, out_path):
    # df = pd.read_csv(raw_path, encoding="ISO-8859-1")
    df = pd.read_csv(raw_path)
    filtered_df = df[df['tweet'].apply(lambda x: len(x) < 512)]

    filtered_df.to_csv(out_path, index=False)
    print(f"Long rows in {raw_path} removed")


def save_with_header(raw_path):
    df = pd.read_csv(raw_path)
    df.columns = ["sentiment", "id", "date", "query", "user", "tweet"]
    df.to_csv(raw_path, index=False)



if __name__ == '__main__':

    unsplit_path = "../data/training_start_to_end.csv"
    header_path = "../data/training_start_to_end_head.csv"
    # unsplit_path = "../data/training.1600000.processed.noemoticon.csv"
    path = "../data/fine_tune"
    path = "../data/fine_tune_bert"
    save_with_header(unsplit_path, header_path)
    remove_long(header_path)

    if len(sys.argv) != 1:
        split(header_path, path, sys.argv[1])
    else:
        split(header_path, path)

