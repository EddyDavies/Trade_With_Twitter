def check_for_duplicates(dictionary_list, item):
    # check no duplicates in list of dictionaries

    items = []
    for dictionary in dictionary_list:
        items.append(dictionary[item])

    if len(items) == len(set(items)):
        return False
    else:
        return True


def append_or_create_list(key: str, container: dict, content: dict):
    if key not in container:
        container[key] = [content]
    else:
        container[key].append(content)
    return container


def to_dict_of_lists(LD):

    nd = {}

    for d in LD:
        try:
            nd['score'].append(d['score'])
        except KeyError:
            nd['score'] = [d['score']]

        try:
            nd['label'].append(d['label'])
        except KeyError:
            nd['label'] = [d['label']]

    return nd