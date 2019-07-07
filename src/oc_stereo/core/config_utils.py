import os

import yaml

import oc_stereo


def config_dict_to_object(config_dict):
    """Recursively converts a dictionary to an object

    Args:
        config_dict: config dictionary to convert

    Returns:
        ConfigObj configuration object
    """

    def convert(item):
        if isinstance(item, dict):
            return type('ConfigObj', (), {k: convert(v) for k, v in item.items()})

        if isinstance(item, list):
            def yield_convert(list_item):
                for index, value in enumerate(list_item):
                    yield convert(value)

            return list(yield_convert(item))
        else:
            return item

    return convert(config_dict)


def no_duplicates_constructor(loader, node, deep=False):
    """Check for duplicate keys
    https://gist.github.com/pypt/94d747fe5180851196eb
    """

    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        if key in mapping:
            raise ValueError('Found duplicate key in yaml' + key)
        mapping[key] = value

    return loader.construct_mapping(node, deep)


def parse_yaml_config(yaml_path):
    """Parses a yaml config

    Args:
        yaml_path: path to yaml config

    Returns:
        config_obj: config converted to object
    """

    # Add check for duplicate keys in yaml
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates_constructor)

    with open(yaml_path, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file)

    config_obj = config_dict_to_object(config_dict)
    config_obj.config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    config_obj.exp_output_dir = oc_stereo.data_dir() + '/outputs/' + config_obj.config_name

    return config_obj
