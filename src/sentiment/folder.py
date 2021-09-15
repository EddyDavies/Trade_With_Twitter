import os


def get_paths(reset=False,
              crypto='bitcoin',
              data_folder='../data',
              model_name=None):
    # ToDo move model name to start and make not kwarg
    # if reset:
    #  ToDo Remove old data

    if model_name:
        model_folder = convert_model_name(model_name)
    else:
        model_folder = "distilbert-base-uncased-finetuned"


    raw_results_folder = f'{crypto}_scores/'
    raw_source_folder = f'{crypto}_tweets/'

    source_folder = os.path.join(data_folder, raw_source_folder)
    results_folder = os.path.join(data_folder, raw_results_folder, model_folder)

    # if not os.path.exists(results_folder):
    #     throw

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    return source_folder, results_folder


def convert_model_name(model_name):
    try:
        model_developer, model_name, = model_name.split('/', 1)
    except:
        pass
    model_folder = '-'.join(model_name.split('-')[:4])
    try:
        model_folder = f"{model_folder}_{model_developer}"
    except:
        pass
    return model_folder