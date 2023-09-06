def flatten_json(json_obj, parent_key='', separator='_'):
    flattened = {}
    for key, value in json_obj.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key, separator=separator))
        elif isinstance(value, list):
            if all(isinstance(item, dict) and 'label' in item and 'score' in item for item in value):
                for item in value:
                    flattened[new_key + '_label'] = str(item['label'])
                    flattened[new_key + '_score'] = str(item['score'])
            else:
                flattened[new_key] = str(value)
        else:
            flattened[new_key] = str(value)
    return flattened