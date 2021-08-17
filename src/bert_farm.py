
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
import logging
import pandas as pd


def read_file(file_name: str) -> dict:
    text_file = open (file_name, 'r')
    text_file = text_file.read().replace('\n', ' ')
    return {'text': text_file}


def create_input(text_files:list) -> list:
    model_input = list()
    for text_file in text_files:
    model_input.append(read_file(text_file['file']))
    return model_input


def create_result_overview (articles:list, result:list) -> pd.DataFrame:
    files = list()
    labels = list()
    predictions = list()
    for i in range(len(articles)):
    files.append (articles[i]['file'])
    labels.append(articles[i]['genre'])
    predictions.append(result[0]['predictions'][i]['label'].strip("'[]'"))
    data = {'file': files, 'actual': labels, 'prediction': predictions}
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':

    # Farm allows simple logging of many parameters & metrics. Let's use the MLflow framework to track our experiment ...
    # You will see your results on https://public-mlflow.deepset.ai/
    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Crypto_Tweet_Bert", run_name="Crypto Tweet BERT")

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 2
    batch_size = 8
    evaluate_every = 100

    lang_model = "bert-base-cased"
    do_lower_case = False

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)


    data_dir = "../data/fine_tune_bert"
    label_list = ['0', '2', '4'] #labels in our data set
    metric = "f1_macro" # desired metric for evaluation

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                max_seq_len=512, # BERT can only handle sequence lengths of up to 512
                                                data_dir=data_dir,
                                                label_list=label_list,
                                                label_column_name="sentiment", # our labels are located in the "genre" column
                                                text_column_name="tweet",
                                                metric=metric,
                                                quote_char='"',
                                                multilabel=True,
                                                delimiter=',',
                                                train_filename="train.csv",
                                                dev_filename=None,
                                                test_filename="test.csv",
                                                dev_split=0.1 # this will extract 10% of the train set to create a dev set
                                                )

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # loading the pretrained BERT base cased model
    language_model = LanguageModel.load(lang_model)
    # prediction head for our model that is suited for classifying news article genres
    prediction_head = MultiLabelTextClassificationHead(num_labels=len(label_list))

    model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm_output_types=["per_sequence"],
            device=device)

    model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=3e-5,
            device=device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs)

    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device)

    trainer.train()

    save_dir = "../data/saved_models/bert"
    model.save(save_dir)
    processor.save(save_dir)

    #
    # inferenced_model = Inferencer.load(save_dir)
    #
    #
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
