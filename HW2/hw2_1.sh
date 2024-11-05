#!/bin/bash

# TODO - run your inference Python3 code
# gdown 1OZsaOeyEIdBeht-bKtbGvTnFzFelawjz -O DLCV_HW2_p1_ckpt.pth
python3 p1_inference.py $1
# python3 digit_classifier.py --folder Output_folder/ --checkpoint ./Classifier.pth