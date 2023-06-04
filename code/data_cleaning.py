import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm


def count_num_of_files():
    letters = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    for l in letters:
        # Directory to check
        directory = "data/train/" + l

        # Get the list of files and directories in the specified directory
        file_list = os.listdir(directory)

        # Count the number of files
        file_count = 0
        for item in file_list:
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                file_count += 1

        # Print the number of files
        print(f"The directory '{directory}' contains {file_count} files.")


if __name__ == '__main__':
    count_num_of_files()
    directory = "data/train/x"
    x_delete = ['afb4e51a-ce5d-11eb-b317-38f9d35ea60f.png',
                'aff0325a-ce5d-11eb-b317-38f9d35ea60f.png',
                'b0156f52-ce5d-11eb-b317-38f9d35ea60f.png',
                'afd7564a-ce5d-11eb-b317-38f9d35ea60f.png',
                'afdd00f4-ce5d-11eb-b317-38f9d35ea60f.png',
                'afc10ba6-ce5d-11eb-b317-38f9d35ea60f.png',
                'afbd2a68-ce5d-11eb-b317-38f9d35ea60f.png',
                'af9e6b64-ce5d-11eb-b317-38f9d35ea60f.png',
                'afc8f028-ce5d-11eb-b317-38f9d35ea60f.png',
                'b02931ea-ce5d-11eb-b317-38f9d35ea60f.png']

    for png_name in x_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/viii"
    viii_delete = ['ad4eb472-ce5d-11eb-b317-38f9d35ea60f.png', 'ad1a9a16-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad9a139a-ce5d-11eb-b317-38f9d35ea60f.png', 'ad1a0948-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad53785e-ce5d-11eb-b317-38f9d35ea60f.png', 'ad9404f0-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad304a78-ce5d-11eb-b317-38f9d35ea60f.png', 'adb07b4e-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad6b1450-ce5d-11eb-b317-38f9d35ea60f.png', 'ad0fcdfc-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad6def90-ce5d-11eb-b317-38f9d35ea60f.png', 'ad18dbae-ce5d-11eb-b317-38f9d35ea60f.png',
                   'ad7dedc8-ce5d-11eb-b317-38f9d35ea60f.png', 'ad7040e2-ce5d-11eb-b317-38f9d35ea60f.png',
                   'adc4777a-ce5d-11eb-b317-38f9d35ea60f.png']

    for png_name in viii_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/vii"
    vii_delete = ['acd12034-ce5d-11eb-b317-38f9d35ea60f.png',
                  'aca544e6-ce5d-11eb-b317-38f9d35ea60f.png',
                  'aca3b2e8-ce5d-11eb-b317-38f9d35ea60f.png',
                  'acbbe7dc-ce5d-11eb-b317-38f9d35ea60f.png',
                  'ac9d42a0-ce5d-11eb-b317-38f9d35ea60f.png',
                  'acdcf062-ce5d-11eb-b317-38f9d35ea60f.png',
                  'ac8ceffe-ce5d-11eb-b317-38f9d35ea60f.png',
                  'acaf1e26-ce5d-11eb-b317-38f9d35ea60f.png',
                  'adc2d0b4-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in vii_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/vi"
    vi_delete = ['ab1d38ae-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aabf4b04-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aac5f63e-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aab65602-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aac26b40-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aaefbfe6-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab1e53ba-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab24bd22-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab291c32-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab23f0b8-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab2356ee-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aac4401e-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aab77122-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in vi_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/v"
    v_delete = ['af4714b8-ce5d-11eb-b317-38f9d35ea60f.png', 'af2f5922-ce5d-11eb-b317-38f9d35ea60f.png',
                'af4fe8b8-ce5d-11eb-b317-38f9d35ea60f.png', 'af5dc294-ce5d-11eb-b317-38f9d35ea60f.png',
                'af88d308-ce5d-11eb-b317-38f9d35ea60f.png', 'af644de4-ce5d-11eb-b317-38f9d35ea60f.png',
                'af7c4732-ce5d-11eb-b317-38f9d35ea60f.png', 'af2fef0e-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in v_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/ix"
    ix_delete = ['aed0b9b2-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aea51e88-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aee14818-ce5d-11eb-b317-38f9d35ea60f.png',
                 'af0f452e-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aeb1e50a-ce5d-11eb-b317-38f9d35ea60f.png',
                 'af03e260-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aec595aa-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aee98348-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aedc86ac-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aeba70da-ce5d-11eb-b317-38f9d35ea60f.png',
                 'af0baa7c-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ae906772-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ae8da37a-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aee2e916-ce5d-11eb-b317-38f9d35ea60f.png',
                 'aee49716-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in ix_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/iv"
    iv_delete = ['ae3362ac-ce5d-11eb-b317-38f9d35ea60f.png', 'ae285998-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ae4fba74-ce5d-11eb-b317-38f9d35ea60f.png', 'adf35f18-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ae0aee76-ce5d-11eb-b317-38f9d35ea60f.png', 'ae507428-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ae482476-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in iv_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/iii"
    iii_delete = ['b0ace94a-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0f00afe-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0729d62-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b03fdcec-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b04248a6-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b02bd5d0-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b03dbf5c-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0492432-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0669cf6-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0356104-ce5d-11eb-b317-38f9d35ea60f.png',
                  'aa8ae80a-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b033b638-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b08ad24c-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0dafcb8-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b058e836-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b04fc576-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b053b2c6-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0ef7d32-ce5d-11eb-b317-38f9d35ea60f.png',
                  'b0dbd0d4-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in iii_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/i"
    i_delete = ['abb304d8-ce5d-11eb-b317-38f9d35ea60f.png', 'ac1fd234-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac5e0554-ce5d-11eb-b317-38f9d35ea60f.png', 'ac225eaa-ce5d-11eb-b317-38f9d35ea60f.png',
                'abe9ea34-ce5d-11eb-b317-38f9d35ea60f.png', 'abe46fb4-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac5baeb2-ce5d-11eb-b317-38f9d35ea60f.png', 'abf4867e-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac3396a2-ce5d-11eb-b317-38f9d35ea60f.png', 'abab984c-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac328dd4-ce5d-11eb-b317-38f9d35ea60f.png', 'ac618bca-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac18fb9e-ce5d-11eb-b317-38f9d35ea60f.png', 'abe3a16a-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac0a17be-ce5d-11eb-b317-38f9d35ea60f.png', 'abe0bb6c-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac59f5b8-ce5d-11eb-b317-38f9d35ea60f.png', 'ac4f0d92-ce5d-11eb-b317-38f9d35ea60f.png',
                'abf55f18-ce5d-11eb-b317-38f9d35ea60f.png', 'ac3e7324-ce5d-11eb-b317-38f9d35ea60f.png',
                'abf6830c-ce5d-11eb-b317-38f9d35ea60f.png', 'ac3db31c-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in i_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    directory = "data/train/ii"
    ii_delete = ['ab58d33c-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab6581d6-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab67a948-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab3962cc-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab75cd48-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab526a6a-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab311ffe-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab5d1a28-ce5d-11eb-b317-38f9d35ea60f.png',
                 'ab4ce810-ce5d-11eb-b317-38f9d35ea60f.png']
    for png_name in ii_delete:
        file_path = os.path.join(directory, png_name)

        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

    count_num_of_files()

    i_keep = ['ab73997e-ce5d-11eb-b317-38f9d35ea60f.png',
              'b042daf0-ce5d-11eb-b317-38f9d35ea60f.png',
              'aefd90f4-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac47f296-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac389e68-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac19def6-ce5d-11eb-b317-38f9d35ea60f.png',
              'abccce40-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac442e68-ce5d-11eb-b317-38f9d35ea60f.png']
    ii_keep = ['b0a8e61a-ce5d-11eb-b317-38f9d35ea60f.png',
               'b0b3a50a-ce5d-11eb-b317-38f9d35ea60f.png',
               'b03f55c4-ce5d-11eb-b317-38f9d35ea60f.png',
               'aebd296a-ce5d-11eb-b317-38f9d35ea60f.png',
               'abf15a58-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac42e6ac-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac013f7c-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac564e2c-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac4b5256-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac315d6a-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac1ea5bc-ce5d-11eb-b317-38f9d35ea60f.png',
               'af71d1f8-ce5d-11eb-b317-38f9d35ea60f.png',
               'adcdd5a4-ce5d-11eb-b317-38f9d35ea60f.png',
               'ad105f2e-ce5d-11eb-b317-38f9d35ea60f.png']
    iii_keep = ['ab713f76-ce5d-11eb-b317-38f9d35ea60f.png',
                'b049dbfc-ce5d-11eb-b317-38f9d35ea60f.png',
                'af0cd44c-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac4be2ca-ce5d-11eb-b317-38f9d35ea60f.png',
                'ae751ff8-ce5d-11eb-b317-38f9d35ea60f.png',
                'ac156ed4-ce5d-11eb-b317-38f9d35ea60f.png',
                'adb27eb2-ce5d-11eb-b317-38f9d35ea60f.png']
    iv_keep = ['b08a2702-ce5d-11eb-b317-38f9d35ea60f.png',
               'aec401ea-ce5d-11eb-b317-38f9d35ea60f.png',
               'af047216-ce5d-11eb-b317-38f9d35ea60f.png',
               'aaba1c60-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab2075be-ce5d-11eb-b317-38f9d35ea60f.png',
               'adb83398-ce5d-11eb-b317-38f9d35ea60f.png',
               'ad7c6d22-ce5d-11eb-b317-38f9d35ea60f.png']
    ix_keep = ['b08eb24a-ce5d-11eb-b317-38f9d35ea60f.png',
               'b04d17c2-ce5d-11eb-b317-38f9d35ea60f.png',
               'adf08b30-ce5d-11eb-b317-38f9d35ea60f.png',
               'ae0745aa-ce5d-11eb-b317-38f9d35ea60f.png']
    v_keep = ['ab48e396-ce5d-11eb-b317-38f9d35ea60f.png',
              'ab5d949e-ce5d-11eb-b317-38f9d35ea60f.png',
              'b05c1808-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac268e44-ce5d-11eb-b317-38f9d35ea60f.png',
              'ae55b49c-ce5d-11eb-b317-38f9d35ea60f.png',
              'adf5ad04-ce5d-11eb-b317-38f9d35ea60f.png',
              'ae12eb1c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af72cc3e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af823a7a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af1e5bae-ce5d-11eb-b317-38f9d35ea60f.png',
              'af815830-ce5d-11eb-b317-38f9d35ea60f.png',
              'af27a506-ce5d-11eb-b317-38f9d35ea60f.png',
              'af773fbc-ce5d-11eb-b317-38f9d35ea60f.png',
              'af7a76a0-ce5d-11eb-b317-38f9d35ea60f.png',
              'af47aec8-ce5d-11eb-b317-38f9d35ea60f.png',
              'af3fbaa6-ce5d-11eb-b317-38f9d35ea60f.png',
              'af4a4e9e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af4d7222-ce5d-11eb-b317-38f9d35ea60f.png',
              'af2da19a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af66b11a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af2c7ebe-ce5d-11eb-b317-38f9d35ea60f.png',
              'af6e3228-ce5d-11eb-b317-38f9d35ea60f.png',
              'af8543fa-ce5d-11eb-b317-38f9d35ea60f.png',
              'af1693b0-ce5d-11eb-b317-38f9d35ea60f.png',
              'af70be76-ce5d-11eb-b317-38f9d35ea60f.png',
              'af25a03a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af448a86-ce5d-11eb-b317-38f9d35ea60f.png',
              'af538e46-ce5d-11eb-b317-38f9d35ea60f.png',
              'af3512e0-ce5d-11eb-b317-38f9d35ea60f.png',
              'af2a3230-ce5d-11eb-b317-38f9d35ea60f.png',
              'af2ea41e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af8710b8-ce5d-11eb-b317-38f9d35ea60f.png',
              'af681280-ce5d-11eb-b317-38f9d35ea60f.png',
              'af62c5a0-ce5d-11eb-b317-38f9d35ea60f.png',
              'af3f2b2c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af7633ce-ce5d-11eb-b317-38f9d35ea60f.png',
              'af37256c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af180114-ce5d-11eb-b317-38f9d35ea60f.png',
              'af34925c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af20072e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af31a0ec-ce5d-11eb-b317-38f9d35ea60f.png',
              'af57414e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af3b0506-ce5d-11eb-b317-38f9d35ea60f.png',
              'af7d78aa-ce5d-11eb-b317-38f9d35ea60f.png',
              'af783d0e-ce5d-11eb-b317-38f9d35ea60f.png',
              'af19717a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af21299c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af5cef54-ce5d-11eb-b317-38f9d35ea60f.png',
              'af580bba-ce5d-11eb-b317-38f9d35ea60f.png',
              'af490642-ce5d-11eb-b317-38f9d35ea60f.png']
    vi_keep = ['ab4d7af0-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab88f44a-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab4e7f90-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab7efbc0-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab327264-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab83e86a-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab7b0d9e-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab868886-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab3826c8-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab79f4e0-ce5d-11eb-b317-38f9d35ea60f.png',
               'b05f2872-ce5d-11eb-b317-38f9d35ea60f.png',
               'b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png',
               'b0735928-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab20f822-ce5d-11eb-b317-38f9d35ea60f.png',
               'ab1bd798-ce5d-11eb-b317-38f9d35ea60f.png',
               'aabbb6b0-ce5d-11eb-b317-38f9d35ea60f.png',
               'abf7a8e0-ce5d-11eb-b317-38f9d35ea60f.png',
               'ac5f0530-ce5d-11eb-b317-38f9d35ea60f.png',
               'ade04432-ce5d-11eb-b317-38f9d35ea60f.png',
               'adfd451e-ce5d-11eb-b317-38f9d35ea60f.png',
               'ae6d4580-ce5d-11eb-b317-38f9d35ea60f.png',
               'af8dac70-ce5d-11eb-b317-38f9d35ea60f.png',
               'af6c0ae8-ce5d-11eb-b317-38f9d35ea60f.png']
    vii_keep = ['aecd5c68-ce5d-11eb-b317-38f9d35ea60f.png']
    viii_keep = []
    x_keep = ['ab5ba788-ce5d-11eb-b317-38f9d35ea60f.png',
              'ab6ae5cc-ce5d-11eb-b317-38f9d35ea60f.png',
              'ae96c6da-ce5d-11eb-b317-38f9d35ea60f.png',
              'aeff0006-ce5d-11eb-b317-38f9d35ea60f.png',
              'aef66fcc-ce5d-11eb-b317-38f9d35ea60f.png',
              'ab1785a8-ce5d-11eb-b317-38f9d35ea60f.png',
              'ab15f102-ce5d-11eb-b317-38f9d35ea60f.png',
              'ac14e8ba-ce5d-11eb-b317-38f9d35ea60f.png',
              'addb9af4-ce5d-11eb-b317-38f9d35ea60f.png',
              'af6dafb0-ce5d-11eb-b317-38f9d35ea60f.png',
              'af54ac2c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af4eb452-ce5d-11eb-b317-38f9d35ea60f.png',
              'af357f8c-ce5d-11eb-b317-38f9d35ea60f.png',
              'af1ddfb2-ce5d-11eb-b317-38f9d35ea60f.png',
              'af6ca11a-ce5d-11eb-b317-38f9d35ea60f.png',
              'af4670e4-ce5d-11eb-b317-38f9d35ea60f.png']

    all_keep_letters = [i_keep, ii_keep, iii_keep, iv_keep, ix_keep, v_keep, vi_keep, vii_keep, viii_keep, x_keep]
    letters = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

    for l_keep, keep in zip(letters, all_keep_letters):
        for png_name in keep:
            for letter in letters:
                if l_keep == letter:
                    continue
                directory = "data/train/" + letter
                file_path = os.path.join(directory, png_name)
                if os.path.exists(file_path):
                    # move the file
                    new_directory = "data/train/" + l_keep
                    new_file_path = os.path.join(new_directory, png_name)
                    os.rename(file_path, new_file_path)

    count_num_of_files()