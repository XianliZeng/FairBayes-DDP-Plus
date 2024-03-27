
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(
        device), torch.FloatTensor(XZ).to(device)




def adult(data_root, display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_train_data = pd.read_csv(
        data_root + 'adult.data',
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    raw_test_data = pd.read_csv(
        data_root + 'adult.test',
        skiprows=1,
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )


    train_data = raw_train_data.drop(["Workclass",'fnlwgt',"Education","Marital Status","Occupation","Relationship","Race","Capital Gain", "Capital Loss","Country"], axis=1)
    test_data = raw_test_data.drop(["Workclass",'fnlwgt',"Education","Marital Status","Occupation","Relationship","Race","Capital Gain", "Capital Loss","Country"], axis=1)
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Workclass",'fnlwgt',"Education","Marital Status","Occupation","Relationship","Race","Capital Gain", "Capital Loss","Country","Target"]), dtypes))


    for k, dtype in filt_dtypes:
        if dtype == "category":

            train_data[k] = train_data[k].cat.codes
            test_data[k] = test_data[k].cat.codes


    train_data["Target"] = train_data["Target"] == " >50K"
    test_data["Target"] = test_data["Target"] == " >50K."

    #
    # all_data = pd.concat([train_data,test_data])
    # data_size = len(all_data)
    # new_index_all = np.arange(data_size)
    # all_data.index = new_index_all

    return train_data,test_data



class CustomDataset_att():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y


class CustomDataset():
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessDataset():
    def __init__(self, dataset,seed, device=torch.device('cuda')):
        self.dataset = dataset
        self.device = device

        if self.dataset == 'AdultCensus':
            self.get_adult_data(seed)
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
        self.prepare_ndarray()

    def get_adult_data(self,seed):
        trainingval_set,testing_set = adult('./data/adult/')
        training_set,validation_set =  train_test_split(trainingval_set, test_size=0.3, random_state=seed)
        training_set.index = np.arange(len(training_set))
        validation_set.index = np.arange(len(validation_set))
        testing_set.index = np.arange(len(testing_set))

        self.Z_train_ = training_set['Sex']
        self.Y_train_ = training_set['Target']
        self.X_train_ = training_set.drop(['Target','Sex'],axis=1)

        self.Z_val_ = training_set['Sex']
        self.Y_val_ = training_set['Target']
        self.X_val_ = training_set.drop(['Target','Sex'],axis=1)
        self.Z_test_ = testing_set['Sex']
        self.Y_test_ = testing_set['Target']
        self.X_test_ = testing_set.drop(['Target','Sex'],axis=1)

        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_val_ = pd.get_dummies(self.X_val_)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(self.Y_train_)
        self.Y_train_ = pd.Series(self.Y_train_, name='>50k')
        self.Y_val_ = le.fit_transform(self.Y_val_)
        self.Y_val_ = pd.Series(self.Y_val_, name='>50k')
        self.Y_test_ = le.fit_transform(self.Y_test_)
        self.Y_test_ = pd.Series(self.Y_test_, name='>50k')


    def prepare_ndarray(self):
        self.normalized = False
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1, 1)], axis=1)
        self.X_val = self.X_val_.to_numpy(dtype=np.float64)
        self.Y_val = self.Y_val_.to_numpy(dtype=np.float64)
        self.Z_val = self.Z_val_.to_numpy(dtype=np.float64)
        self.XZ_val = np.concatenate([self.X_val, self.Z_val.reshape(-1, 1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1, 1)], axis=1)

        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def normalize(self):
        self.normalized = True
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_val = scaler_XZ.fit_transform(self.XZ_val)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)

        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_val = scaler_X.fit_transform(self.X_val)
        self.X_test = scaler_X.transform(self.X_test)
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),  \
               (self.X_val, self.Y_val, self.Z_val, self.XZ_val),  \
                (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_val_, Y_val_, Z_val_, XZ_val_ = arrays_to_tensor(
            self.X_val, self.Y_val, self.Z_val, self.XZ_val, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_), \
               (X_val_, Y_val_, Z_val_, XZ_val_), \
               (X_test_, Y_test_, Z_test_, XZ_test_)