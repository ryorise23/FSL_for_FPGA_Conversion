"""
neural network modules
handle loading, inference and prediction
"""
import numpy as np
from typing import Union, Sequence

from few_shot_model.numpy_utils import softmax, one_hot, k_small

from icecream import ic
"""
def softmax(x: np.ndarray, dim=0):

    # stability trick( cond of exp(x)=x )
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)
"""

def feature_preprocess(features: np.ndarray, mean_base_features: np.ndarray):
    """
    preprocess the feature (normalisation on the unit sphere) for classification
    Args :
        features(np.ndarray) : feature to be preprocessed
        mean_base_features(np.ndarray) : expected mean of the tensor
    returns:
        features(np.ndarray) : normalized feature
    """
    features = features - mean_base_features #features.shape: (20, 5, 80), mean_base_features.shape: (20, 1, 80)
    features = features / np.linalg.norm(features, axis=-1, keepdims=True) #features.shape: (20, 5, 80)
    return features


def ncm(shots_mean: np.ndarray, features: np.ndarray):
    """
    compute the class attribution probas using the ncm classifier
    args :
        - shots_mean array(...,n_class,n_dim) : mean of the saved shots for each classe
        - features array(...,n_dim) : features to classify (leading dims same as previous array)
    """
    
    features = np.expand_dims(features, axis=-2)  # broadcastable along class axis, #features.shape: (20, 5, 15, 1, 80)
    distances = np.linalg.norm(shots_mean - features, axis=-1, ord=2) #shots_mean.shape: (20, 1, 1, 5, 80), distances.shape: (20, 5, 15, 5)
    probas = softmax(-20 * distances, dim=-1) #probas.shape: (20, 5, 15, 5)
    return probas

###行列積による実装例
def compute_distance_matrix(batch_features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    バッチ内の各特徴ベクトルと各クラスのプロトタイプ間の2乗距離を計算する。
    batch_features : shape (N, D)  - N個のサンプル、各サンプルの次元はD
    prototypes     : shape (C, D)  - Cクラス分のプロトタイプ
    return         : shape (N, C)  - 各サンプルと各クラス間の2乗距離
    """
    # 各サンプルの2乗ノルム (N, 1)
    X_norm = np.sum(batch_features ** 2, axis=1, keepdims=True)
    # 各プロトタイプの2乗ノルム (C, 1)
    P_norm = np.sum(prototypes ** 2, axis=1, keepdims=True)
    # 内積の計算 (N, C)
    inner_product = np.dot(batch_features, prototypes.T)
    # ブロードキャストを利用して2乗距離を計算
    distances_sq = X_norm + P_norm.T - 2 * inner_product
    # 数値誤差対策で、負の値を0に
    distances_sq = np.maximum(distances_sq, 0)
    return distances_sq

def ncm_batch_predict(batch_features: np.ndarray, prototypes: np.ndarray, temperature: float = 20.0):
    """
    バッチ処理でNCM分類を行い、各サンプルのクラス確率と予測クラスを返す。
    batch_features : shape (N, D)
    prototypes     : shape (C, D)
    temperature    : softmaxのスケール（大きくすると判別が鋭くなる）
    """
    # 2乗距離を計算（ここでは距離の大小で順位付けできれば、平方根をとる必要はありません）
    distances_sq = compute_distance_matrix(batch_features, prototypes)
    # 実際のユークリッド距離が必要な場合は下記のようにsqrtする
    distances = np.sqrt(distances_sq)
    
    # 距離に対してスケールをかけた値の負の値を logits として softmax を計算
    logits = -temperature * distances
    # 数値安定化のために各行の最大値を引く
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 各サンプルで最も高い確率のクラスを予測とする
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities
###


def knn(
    shots_points: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    number_neighboors: int,
):
    """
    compute the class attribution probas using the ncm classifier
    args :
        - shots_mean array(...,n_points,n_dim) : mean of the saved shots for each classe
        - features array(...,n_dim) : features to classify (leading dims same as previous array)
        - target : array(n_points) : represent feature assignement. Expected to have value in [0, ...,n_class-1]
        - number_neighboors (int) : number of neighboors to take
    """
    number_class = np.max(target) + 1

    features = np.expand_dims(features, axis=-2)  # broadcastable along point axis
    distances = np.linalg.norm(shots_points - features, axis=-1, ord=2)

    indices = k_small(distances, number_neighboors, axis=-1)

    probas = one_hot(target[indices], number_class)

    # mean along neighboors
    probas = np.sum(probas, axis=-2) / number_neighboors

    return probas


class FewShotModel:
    """
    class defining a few shot model (A model predicting the class from the feature using the references features)
    attributes :
        - backbone : initialized with backbone_specs(dict):
            specs defining the how to load the backbone
        - classifier_specs :
            parameters of the final classification model
        - preprocess : how preprocess input image
        - device : on wich device should the computation take place

    2 possible types of prediction (batchified or not)
    """

    def __init__(self, classifier_specs: dict):
        self.classifier_specs = classifier_specs

    def predict_class_batch(
        self,
        features: np.ndarray,
        shot_array: np.ndarray,
        mean_feature: np.ndarray,
        preprocess_feature=True,
    ):
        """
        predict the class of a features
        args:
            features :
                - (np.ndarray(n_batch,nways,n_queries,n_features)) : features of the current img
            shot_array :
                - array(n_batch,n_ways,n_shots,n_features) (each element of sequence = 1 class)
            mean_feature :
                - array(n_batch,n_features)
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas (1,n_features) : probability of belonging to each class

        """
        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs.get("kwargs", {})
        # shots_list = recorded_data.get_shot_list()

        if preprocess_feature:
            # (n_batch,1,1,n_features)
            features = feature_preprocess(
                features, np.expand_dims(mean_feature, axis=(1, 2))
            )

        # class asignement using the correspounding model

        if model_name == "ncm":
            shots = np.mean(shot_array, axis=2)  # mean of the shots
            # (n_batch,n_ways,n_features)
            # shots=shots.detach().cpu().numpy()
            if preprocess_feature:
                # (n_batch,1,n_features)
                shots = feature_preprocess(shots, np.expand_dims(mean_feature, axis=1))
            shots = np.expand_dims(shots, axis=(1, 2))
            # (_batch,1,1,n_ways,n_features)

            import time
            #t = time.time()
            probas = ncm(shots, features)
            #ic(time.time()-t)

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            # create target list of the shots
            n_ways = shot_array.shape[1]
            n_shots = shot_array.shape[2]
            shots = np.reshape(
                shot_array,
                axis=(shot_array.shape[0], n_ways * n_shots, shot_array.shape[3]),
            )
            # shots : (n_batch,n_exemples,nfeatures)
            if preprocess_feature:
                shots = feature_preprocess(shots, np.expand_dims(mean_feature, axis=1))
            shots = np.expand_dims(shots, axis=(2, 3))
            # (_batch,n_ways,1,n_features)

            targets = np.concatenate(
                [
                    class_id * np.ones(n_shots, dtype=np.int64)
                    for class_id in range(n_ways)
                ],
                axis=0,
            )

            probas = knn(shots, features, targets, number_neighboors)        

        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        classe_prediction = np.argmax(probas, axis=-1)
        return classe_prediction, probas

    def predict_class_feature(
        self,
        features: np.ndarray,
        shots_list: Sequence[np.ndarray],
        mean_feature: np.ndarray,
        preprocess_feature=True,
    ):
        """
        predict the class of a features

        args:
            features :
                - (np.ndarray(n_features)) : features of the current img
            shot_list :
                - sequence(array(n_shots_i,n_features)) (each element of sequence = 1 class)
            mean_feature :
                - array(n_features)
            model_name : wich model do we use
            **kwargs : additional parameters of the model
        returns :
            classe_prediction : class prediction
            probas (1,n_features) : probability of belonging to each class
        """

        # mean_feature = np.mean(shots_list,axis=0) #recorded_data.get_mean_features()

        model_name = self.classifier_specs["model_name"]
        model_arguments = self.classifier_specs.get("kwargs", {})
        # shots_list = recorded_data.get_shot_list()

        if preprocess_feature:
            features = feature_preprocess(features, mean_feature)

        # class asignement using the correspounding model

        if model_name == "ncm":
            shots = np.stack(
                [np.mean(shot, axis=0) for shot in shots_list], axis=0 #shot分はここで平均を求めているから[N, 80]となる(N:クラス数)
            )  # sequence -> array
            # shots : (nclass,nfeatures)
            # shots=shots.detach().cpu().numpy()
            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)
            probas = ncm(shots, features)

        elif model_name == "knn":
            number_neighboors = model_arguments["number_neighboors"]
            number_samples_class_1 = shots_list[0].shape[0]
            for shot in shots_list:
                assert (
                    shot.shape[0] == number_samples_class_1
                ), "knn requires an even number of samples per class"

            # sequence -> array
            shots = np.concatenate(shots_list, axis=0)
            # shots : (n_exemples, nfeatures)

            if preprocess_feature:
                shots = feature_preprocess(shots, mean_feature)

            number_ways = len(shots_list)  # ok for sequence and array

            targets = np.concatenate(
                [
                    class_id * np.ones(shots_list[class_id].shape[0], dtype=np.int64)
                    for class_id in range(number_ways)
                ],
                axis=0,
            )

            probas = knn(shots, features, targets, number_neighboors)

        else:
            raise NotImplementedError(f"classifier : {model_name} is not implemented")

        classe_prediction = np.argmax(probas, axis=-1)
        return classe_prediction, probas

    def predict_class_moving_avg(
        self,
        features: np.ndarray,
        prev_probabilities: Union[None, np.ndarray],
        shots_list: Sequence[np.ndarray],
        mean_feature: np.ndarray,
    ):
        """

        update the probabily and attribution of having a class, using the current image
        args :
            features(np.ndarray((1,n_features))) : features of the current img
            prev_probabilities(?) : probability of each class for previous prediction
            recorded_data (DataFewShot) : data recorded for classification

        returns :
            classe_prediction : class prediction
            probas : probability of belonging to each class
        """
        model_name = self.classifier_specs["model_name"]

        _, current_proba = self.predict_class_feature(
            features, shots_list, mean_feature
        )


        if prev_probabilities is None:
            probabilities = current_proba
        else:
            if model_name == "ncm":
                probabilities = prev_probabilities * 0.85 + current_proba * 0.15
            elif model_name == "knn":
                probabilities = prev_probabilities * 0.95 + current_proba * 0.05

        classe_prediction = probabilities.argmax()
        return classe_prediction, probabilities


import time
row = [1,2,4,8,16,32,64,128]
col = [40,80,160,320,640]
test_num_class = 1
for i in row:
    for j in col:
        test_input = np.random.rand(1, i, j).astype(np.float32)
        test_shot = np.random.rand(test_num_class, i, j).astype(np.float32)
        #a = time.time()
        ncm(test_shot, test_input)
        #ic(time.time() - a)
