"""
"""
import time
import numpy as np

# from memory_profiler import profile
from args import get_args_evaluation
from performance_evaluation.few_shot_eval import get_features_numpy
from performance_evaluation.dataset_numpy import get_dataset_numpy
from backbone_loader.backbone_loader import get_model
from performance_evaluation.few_shot_eval import (
    define_runs,
)
from few_shot_model.few_shot_model import FewShotModel

from icecream import ic

# @profile#comment/uncoment and flag -m memory_profiler after python


def evaluate_model(backbone, kwargs: dict):
    assert kwargs.sample_per_class % kwargs.batch_size == 0

    data = get_dataset_numpy(kwargs.dataset_path)  #

    """
    # データセットをNumPy配列として読み込む（例：imagesがデータセット全体）
    images = np.array(data)

    # 各チャネルの平均と標準偏差を計算
    mean = np.mean(images, axis=(0, 1, 2))  # 各チャネルの平均
    std = np.std(images, axis=(0, 1, 2))    # 各チャネルの標準偏差

    # 統計量をファイルに保存
    with open("data_statistics_pt.txt", "w") as f:
        f.write(f"Mean: {mean.tolist()}\n")
        f.write(f"Standard Deviation: {std.tolist()}\n")
    # データをNumPy形式で保存
    np.save("images_data_pt.npy", data)
    """

    num_classes_data, num_exemples_data, h, w, c = np.shape(data)

    # check compatibility of arguments
    assert num_classes_data >= kwargs.num_classes
    assert num_exemples_data >= kwargs.sample_per_class

    # subset of the data if needed
    data = data[0 : kwargs.num_classes, 0 : kwargs.sample_per_class, :, :, :]

    #normalization (assume the backbone used miniimagnet normalisation during training)
    #data = (data / 255 - np.array([0.485, 0.456, 0.406], dtype=data.dtype)) / np.array(
    #    [0.229, 0.224, 0.225], dtype=data.dtype
    #)
    data = (data / 255 - np.array([0.4720, 0.4536, 0.4094], dtype=data.dtype)) / np.array(
        [0.2767, 0.2692, 0.2852], dtype=data.dtype)
    
    #cifar-10 norm
    #data = (data / 255 - np.array([0.4914, 0.4822, 0.4465], dtype=data.dtype)) / np.array(
    #   [0.2470, 0.2435, 0.2616], dtype=data.dtype)
    #data = np.round(data)

    #data = data / 255
    
    # データの範囲を取得
    min_val = np.min(data)
    max_val = np.max(data)

    int8 = -1
    if int8 == 0:
        # データ範囲を確認
        min_value, max_value = data.min(), data.max()

        # スケーリングファクターを計算
        scale_factor = 127 / max(abs(min_value), abs(max_value))
        print(f"Scale factor: {scale_factor}")

        # スケーリングと量子化
        data = (data * scale_factor).astype(np.int8).astype(np.float32)
        #スケーリングを戻す
        data = data / 48
        #print("Scaled integer data:", data)

    elif int8 == 1:
        # データ範囲を確認
        min_value, max_value = data.min(), data.max()
        print(f"Data range: min={min_value}, max={max_value}")

        # スケーリングファクターを計算
        scale_factor = 127 / max(abs(min_value), abs(max_value))
        print(f"Scale factor: {scale_factor}")

        # スケーリングと量子化
        data = (data * scale_factor).astype(np.int8)
        # 型をfloat32にキャストし、Tensorに変換
        data = data.astype(np.float32)
        import torch
        #data = torch.tensor(data).to('cuda')

        # スケーリング後の範囲を確認
        print(f"Scaled data range: min={data.min()}, max={data.max()}")

    elif int8 == 2:
        data = data * 10000000000
        data = np.round(data)
        data = data.astype(np.float32)
        print(data)

    elif int8 == 3:
        def float_to_fixed_8_8_array(arr: np.ndarray) -> np.ndarray:
            # スケーリングして整数に変換
            fixed_arr = np.round(arr * 256).astype(np.int16)  # 2^8 = 256, 16ビット整数
            # 範囲をクリップして16ビットの範囲に収める
            fixed_arr = np.clip(fixed_arr, -32768, 32767)  # 16ビット符号付き整数の範囲
            return fixed_arr
        def fixed_to_float_8_8_array(fixed_arr: np.ndarray) -> np.ndarray:
            # 固定小数点から浮動小数点に変換
            return fixed_arr / 256.0  # 2^8 = 256
        data = float_to_fixed_8_8_array(data)
        data = fixed_to_float_8_8_array(data)
        data = data.astype(np.float32)
    
    elif int8 == 4: # 非線形スケーリング（ヒストグラム均等化）
        from skimage import exposure

        # CDFに基づいたヒストグラム均等化
        data = exposure.equalize_hist(data)
        data = (data * 255 - 128).astype(np.int8)
        data = data.astype(np.float32)
        # スケーリング後の範囲を確認
        print(f"Scaled data range: min={data.min()}, max={data.max()}")

    elif int8 == 5: #動的範囲スケーリング（対称スケーリング）
        data_centered = data - data.mean()
        scale_factor = 127 / max(abs(data_centered.min()), abs(data_centered.max()))
        data = (data_centered * scale_factor).astype(np.int8)
        data = data.astype(np.float32)
        # スケーリング後の範囲を確認
        print(f"Scaled data range: min={data.min()}, max={data.max()}")

    elif int8 == 6:
        median = np.median(data)
        min_offset, max_offset = data.min() - median, data.max() - median
        scale_factor = 127 / max(abs(min_offset), abs(max_offset))
        data = ((data - median) * scale_factor).clip(-128, 127).astype(np.int8)
        data = data.astype(np.float32)
        # スケーリング後の範囲を確認
        print(f"Scaled data range: min={data.min()}, max={data.max()}")
    
    elif int8 == 7: 
        # データの符号を維持するため、絶対値の対数スケーリングを適用し、再符号化する
        # 負の値にも対応するための対数スケーリング
        sign = np.sign(data)  # 元データの符号を保持
        data_log_scaled = np.log1p(np.abs(data)) * sign

        # 対数スケール後のデータ範囲を確認
        min_log, max_log = data_log_scaled.min(), data_log_scaled.max()
        print(f"Log scaled data range: min={min_log}, max={max_log}")

        # -128~127の範囲に収めるためのスケーリングファクターを計算
        scale_factor = 127 / max(abs(min_log), abs(max_log))
        print(f"Scale factor for int8: {scale_factor}")

        # スケーリングと量子化
        data = (data_log_scaled * scale_factor).astype(np.int8)
        data = data.astype(np.float32)
        # スケーリング後の範囲を確認
        print(f"Scaled data range: min={data.min()}, max={data.max()}")

    # データセットをNumPy配列として読み込む（例：imagesがデータセット全体）
    images = np.array(data)

    # 各チャネルの平均と標準偏差を計算
    mean = np.mean(images, axis=(0, 1, 2))  # 各チャネルの平均
    std = np.std(images, axis=(0, 1, 2))    # 各チャネルの標準偏差

    # 統計量をファイルに保存
    #with open("data_statistics_scaled.txt", "w") as f:
    #    f.write(f"Mean: {mean.tolist()}\n")
    #    f.write(f"Standard Deviation: {std.tolist()}\n")
    # データをNumPy形式で保存
    np.save("images_data_scaled_1203.npy", data)
    #print(data)
    print("range min:", np.min(data), "max:", np.max(data))

    seconds = time.time()
    features = get_features_numpy(backbone, data, kwargs.batch_size)
    
    # データ範囲を確認
    min_value, max_value = features.min(), features.max()
    print(f"Data range: min={min_value}, max={max_value}")
    """
    # スケーリングファクターを計算
    scale_factor = 127 / max(abs(min_value), abs(max_value))
    print(f"Scale factor: {scale_factor}")

    # スケーリングと量子化
    features = (features * scale_factor).astype(np.int8)

    # スケーリング後の範囲を確認
    print(f"Scaled data range: min={features.min()}, max={features.max()}")
    """
    #features = features * 0.0078125
    #features = np.load("feat.npy")
    #noise = np.random.normal(0, 0.01, features.shape)  # 平均0、標準偏差0.01のノイズを生成
    #features = features + noise
    #min_val = np.min(features)
    #max_val = np.max(features)

    #print("range min:", np.min(features), "max:", np.max(features))
    #print("output", features)
    dt_inference = time.time() - seconds

    total_samples = kwargs.num_classes * kwargs.sample_per_class
    mean_speed = dt_inference / total_samples

    # sample_per_class=600
    classe, index = define_runs(
        kwargs.n_runs,
        kwargs.n_ways,
        kwargs.n_shots,
        kwargs.n_queries,
        kwargs.num_classes,
        [kwargs.sample_per_class] * kwargs.num_classes,
    )
    # cifar10 : 122mb
    # runs : 84kb

    index_shots, index_queries = (
        index[:, :, : kwargs.n_shots],
        index[:, :, kwargs.n_shots :],
    )
    extracted_shots = features[              #extracted_shots.shape: (10000, 5, 5, 80)
        np.stack([classe] * kwargs.n_shots, axis=-1), index_shots
    ]  # compute features corresponding to each experiment shots
    extracted_queries = features[            #extracted_queries.shape: (10000, 5, 15, 80)
        np.stack([classe] * kwargs.n_queries, axis=-1), index_queries
    ]  # compute features corresponding to each experiment queries

    mean_feature = np.mean(extracted_shots, axis=(1, 2)) #mean_feature.shape: (10000, 80)

    bs = kwargs.batch_size_fs
    fs_model = FewShotModel(kwargs.classifier_specs)
    perf = []

    for i in range(kwargs.n_runs // bs):
        # view, no data
        batch_q = extracted_queries[i * bs : (i + 1) * bs] #batch_q.shape: (20, 5, 15, 80)
        batch_shot = extracted_shots[i * bs : (i + 1) * bs] #batch_shot.shape: (20, 5, 5, 80)
        batch_mean_feature = mean_feature[i * bs : (i + 1) * bs] #batch_mean_feature.shape: (20, 80)

        predicted_class, _ = fs_model.predict_class_batch(  #predicted_class.shape: (20, 5, 15)
            batch_q, batch_shot, batch_mean_feature
        )
        perf.append(
            np.mean(predicted_class == np.expand_dims(np.arange(0, 5), axis=(0, 2)))
        )
    return np.mean(perf), np.std(perf), mean_speed


def launch_evaluation(kwargs: dict):
    """
    launch a evalution using feature a namespace kwargs, with attributes :
        - backbone_specs
        - dataset_path
        - num_classes_dataset
        - batch_size
        - num_classes
        - sample_per_class
        - n_shots
        - n_ways
        - n_queries
        - batch_size_fs
        - classifier_specs

    """
    # from lim_ram import set_limit
    backbone = get_model(kwargs.backbone_specs)

    return evaluate_model(backbone, kwargs)


if __name__ == "__main__":
    # set_limit(500*1024*1024)#simulate the memmory limitation of the pynk
    mean, std, mean_speed = launch_evaluation(get_args_evaluation())
    print(f"perf : {mean}, +-{std}")
    print(f"avg speed  : {mean_speed} s")
