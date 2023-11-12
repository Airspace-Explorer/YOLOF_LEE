import os
import requests
import yaml
import glob as glob
from mmdet.apis import init_detector
def download_weights(url, file_save_name):
    """
    Download weights for any model.
    :param url: Download URL for the weihgt file.
    :param file_save_name: String name to save the file on to disk.
    """
    # Make chekcpoint directory if not presnet.
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    # Download the file if not present.
    if not os.path.exists(os.path.join('checkpoint', file_save_name)):
        file = requests.get(url)
        open(
            os.path.join('checkpoint', file_save_name), 'wb'
        ).write(file.content)

def parse_meta_file():
    root_meta_file_path = './data/metafile/'
    all_metal_file_paths = glob.glob(os.path.join(root_meta_file_path, '*', 'metafile.yml'), recursive=True)
    
    all_models = []

    for meta_file_path in all_metal_file_paths:
        with open(meta_file_path, 'r') as file:
            yaml_file = yaml.load(file, Loader=yaml.FullLoader)

        if 'Models' in yaml_file and isinstance(yaml_file['Models'], list):
            all_models.extend(yaml_file['Models'])
    
    return all_models

def get_model(weights_name):
    """
    Either downloads a model or loads one from local path if already
    downloaded using the weight file name (`weights_name`) provided.
    :param weights_name: Name of the weight file. Most like in the format
        retinanet_ghm_r50_fpn_1x_coco. SEE `weights.txt` to know weight file
        name formats and downloadable URL formats.
    Returns:
        model: The loaded detection model.
    """

def get_model(config_file,weights_path):
    """
    Loads a model from the local path using the provided weights file path.
    :param weights_path: Full path to the weights file.
    Returns:
        model: The loaded detection model.
    """
    assert os.path.exists(weights_path), f"{weights_path} weight file not found!!!"
    model = init_detector(config=config_file, checkpoint=weights_path, device='cuda:0')
    return model


def write_weights_txt_file():
    """
    Write all the model URLs to `weights.txt` to have a complete list and
    choose one of them.
    EXECUTE `utils.py` if `weights.txt` not already present.
    `python utils.py` command will generate the latest `weights.txt`
    file according to the cloned mmdetection repository.
    """
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()
    with open('weights.txt', 'w') as f:
        for weights in weights_list:
            f.writelines(f"{weights}\n")
    f.close()
if __name__ == '__main__':
    write_weights_txt_file()
    weights_list = parse_meta_file()
    print(weights_list[:3])
