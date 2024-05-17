import pandas as pd
import torch
import ast

COLUMNS = ["scores", "low_score", "high_score", "input_ids", "input_tokens", "subject_range", "answer", "window", "kind"]

def read_all_flow_data(file_path):
    all_flow_data = []
    df = pd.read_csv(file_path)
    # 各行を一行ずつ処理
    for index, row in df.iterrows():
        flow_data = []
        for column in COLUMNS:
            if column in ["low_score", "answer", "window", "kind"]:
                flow_data.append(row[column])
            elif "tensor" in row[column]:
                flow_data.append(string_to_tensor(row[column]))
            else:
                flow_data.append(ast.literal_eval(row[column]))
        dictionary = dict(zip(COLUMNS, flow_data))
        all_flow_data.append(dictionary)
    return all_flow_data

def convert_to_float_array(string_array):
    """
    Convert a string representation of a 1D, 2D array, single float, or a float in scientific notation to an actual 
    array of floats or single float. Handles single float, floats in scientific notation, 1D arrays, and 2D arrays.

    :param string_array: A string representing a single float, 1D or 2D array of floats, or a float in scientific notation.
    :return: A float, 1D array of floats, or 2D array of floats.
    """
    # Check if the string represents a single float or a float in scientific notation
    try:
        # Attempt to convert directly to a float
        return float(string_array)
    except ValueError:
        # Not a single float or a float in scientific notation, proceed to check for arrays
        pass

    # Check if the string represents a 2D array (nested arrays)
    if "],[" in string_array:
        # Process as 2D array
        split_arrays = string_array.split("],[")
        return [[float(num) for num in arr.strip("[]").split(",")] for arr in split_arrays]
    else:
        # Process as 1D array
        return [float(num) for num in string_array.strip("[]").split(",")]

def string_to_tensor(tensor_string):
    # 文字列のノイズを除去
    tensor_string = tensor_string.replace('\n', "")
    tensor_string = tensor_string.replace(' ', "")
    # "tensor(" と ")" を取り除く
    tensor_string = tensor_string.replace("tensor(", "").replace(")", "")
    # "device='cuda:0'" を取り除く
    tensor_string = tensor_string.replace(",device='cuda:0'", "")
    # floatの配列に変換
    float_arrays = convert_to_float_array(tensor_string)
    # テンソルに変換
    tensor = torch.tensor(float_arrays)
    return tensor

# 例
# tensor_string = "tensor([[0.0011, 0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009], [0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0008], [0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0008]], device='cuda:0')"
# tensor_string = "tensor([0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0008], device='cuda:0')"
# tensor_string = """tensor([[7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05],
#         [7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05],
#         [7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05,
#          7.1915e-05, 7.1915e-05, 7.1915e-05, 7.1915e-05],
#         [4.6532e-05, 4.6408e-05, 5.1418e-05, 5.4563e-05, 5.0922e-05, 5.4431e-05,
#          5.5976e-05, 6.9303e-05, 6.6686e-05, 7.0246e-05, 6.7831e-05, 6.7996e-05,
#          6.8828e-05, 7.2319e-05, 7.2137e-05, 7.6374e-05, 7.6339e-05, 7.2073e-05,
#          7.1492e-05, 7.9842e-05, 7.5870e-05, 8.0270e-05, 7.7556e-05, 7.3676e-05,
#          6.9462e-05, 6.5447e-05, 7.2241e-05, 7.1915e-05],
#         [7.6886e-05, 8.4850e-05, 9.1347e-05, 8.6647e-05, 8.3450e-05, 7.9128e-05,
#          7.9042e-05, 8.2583e-05, 8.8956e-05, 8.1822e-05, 7.6194e-05, 7.3122e-05,
#          7.1488e-05, 7.0292e-05, 6.6886e-05, 6.7202e-05, 6.5409e-05, 6.7127e-05,
#          6.5099e-05, 6.4243e-05, 6.4354e-05, 6.4612e-05, 6.5826e-05, 7.2615e-05,
#          7.0584e-05, 7.0642e-05, 7.2182e-05, 7.1915e-05],
#         [7.5575e-05, 9.7352e-05, 9.5017e-05, 9.4026e-05, 8.6680e-05, 8.9941e-05,
#          9.4962e-05, 8.4589e-05, 8.6893e-05, 8.0621e-05, 7.8838e-05, 7.7529e-05,
#          7.7473e-05, 7.8103e-05, 7.7043e-05, 7.7848e-05, 7.5578e-05, 7.5212e-05,
#          7.3996e-05, 7.5443e-05, 7.5574e-05, 7.7142e-05, 7.4585e-05, 7.3020e-05,
#          7.2167e-05, 7.2644e-05, 7.2370e-05, 7.1915e-05],
#         [9.0011e-05, 1.0747e-04, 1.1605e-04, 1.3251e-04, 1.3451e-04, 1.2690e-04,
#          1.3107e-04, 1.6978e-04, 1.2875e-04, 1.2595e-04, 1.2728e-04, 1.1715e-04,
#          1.1103e-04, 1.1275e-04, 9.1811e-05, 8.8247e-05, 8.1372e-05, 7.8459e-05,
#          7.7513e-05, 7.2880e-05, 7.4084e-05, 7.4255e-05, 7.3899e-05, 7.4688e-05,
#          7.4351e-05, 7.4659e-05, 7.2277e-05, 7.1915e-05],
#         [6.6786e-05, 6.4180e-05, 7.1572e-05, 7.5371e-05, 8.3446e-05, 9.5072e-05,
#          8.8701e-05, 9.7342e-05, 7.9253e-05, 8.0123e-05, 8.0345e-05, 8.4775e-05,
#          8.5341e-05, 8.2329e-05, 8.3304e-05, 7.1423e-05, 7.2820e-05, 7.1560e-05,
#          7.1549e-05, 7.0579e-05, 7.1034e-05, 7.1115e-05, 7.0632e-05, 7.0491e-05,
#          7.1111e-05, 7.1013e-05, 7.1471e-05, 7.1915e-05],
#         [7.1632e-05, 6.8282e-05, 6.6220e-05, 6.9024e-05, 7.2889e-05, 8.4175e-05,
#          8.3479e-05, 8.2366e-05, 8.2176e-05, 8.3574e-05, 7.6605e-05, 7.7090e-05,
#          7.6784e-05, 7.6489e-05, 7.5100e-05, 7.3308e-05, 7.4307e-05, 7.3789e-05,
#          7.2604e-05, 7.2475e-05, 7.3201e-05, 7.2314e-05, 7.1947e-05, 7.1959e-05,
#          7.1912e-05, 7.2132e-05, 7.0907e-05, 7.1915e-05],
#         [9.5092e-05, 9.3788e-05, 8.0930e-05, 8.3412e-05, 9.8217e-05, 9.0138e-05,
#          8.7132e-05, 8.2747e-05, 1.0484e-04, 7.9245e-05, 8.9585e-05, 8.2477e-05,
#          8.5136e-05, 8.6641e-05, 9.8104e-05, 1.0764e-04, 1.1892e-04, 1.1848e-04,
#          1.2438e-04, 1.4723e-04, 1.3665e-04, 1.1534e-04, 1.0739e-04, 9.9376e-05,
#          9.8560e-05, 8.8192e-05, 9.4433e-05, 9.4214e-05]])"""
# tensor = string_to_tensor(tensor_string)
# print(tensor)

# df = read_all_flow_data("data/all_flow_data/EleutherAI_gpt-j-6B.csv")
# print(df)