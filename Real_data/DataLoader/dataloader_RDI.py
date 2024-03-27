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


    train_data = raw_train_data.drop(["Education",'fnlwgt'], axis=1)  # redundant with Education-Num
    test_data = raw_test_data.drop(["Education",'fnlwgt'], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))

    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                train_data[k] = np.array([rcode[v.strip()] for v in train_data[k]])
                test_data[k] = np.array([rcode[v.strip()] for v in test_data[k]])
            else:
                train_data[k] = train_data[k].cat.codes
                test_data[k] = test_data[k].cat.codes


    train_data["Target"] = train_data["Target"] == " >50K"
    test_data["Target"] = test_data["Target"] == " >50K."


    all_data = pd.concat([train_data,test_data])
    data_size = len(all_data)
    new_index_all = np.arange(data_size)
    all_data.index = new_index_all


    return all_data



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
        elif self.dataset == 'COMPAS':
            self.get_compas_data(seed)
        elif self.dataset == 'Lawschool':
            self.get_lawschool_data(seed)
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))

    def get_adult_data(self,seed):
        alldata = adult('./data/adult/')
        alldata['Sensitive']=alldata['Sex']
        alldata = alldata.drop(['Sex'],axis = 1)
        self.training_set, self.testing_set = train_test_split(alldata, test_size=0.3, random_state=seed)



    def get_dataset(self, validation=False, val_portion=.0):

        return (self.training_set,self.testing_set)




def Get_data_tensor(dataset_name,training_set,testing_set,Y_train, Y_test,device):

    if dataset_name == 'AdultCensus':
        training_set.index = np.arange(len(training_set))
        testing_set.index = np.arange(len(testing_set))

        Z_train_ = training_set['Sensitive']
        X_train_ = training_set.drop([ 'Sensitive'], axis=1)
        Y_train_ = Y_train.copy()


        Z_test_ = testing_set['Sensitive']
        X_test_ = testing_set.drop([ 'Sensitive'], axis=1)
        Y_test_ = Y_test.copy()

        X_train_ = pd.get_dummies(X_train_)

        X_test_ = pd.get_dummies(X_test_)

        le = LabelEncoder()
        Y_train_ = le.fit_transform(Y_train_)
        Y_train_ = pd.Series(Y_train_, name='Tarket')
        Y_test_ = le.fit_transform(Y_test_)
        Y_test_ = pd.Series(Y_test_, name='Tarket')



    if dataset_name == 'AdultCensus':
        X_train = X_train_.to_numpy(dtype=np.float64)
        Y_train = Y_train_.to_numpy(dtype=np.float64)
        Z_train = Z_train_.to_numpy(dtype=np.float64)


        X_test = X_test_.to_numpy(dtype=np.float64)
        Y_test = Y_test_.to_numpy(dtype=np.float64)
        Z_test = Z_test_.to_numpy(dtype=np.float64)


    XZ_train = np.concatenate([X_train, Z_train.reshape(-1, 1)], axis=1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(-1, 1)], axis=1)




    scaler_XZ = StandardScaler()
    XZ_train = scaler_XZ.fit_transform(XZ_train)
    XZ_test = scaler_XZ.transform(XZ_test)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)


    X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
        X_train, Y_train, Z_train, XZ_train, device)
    X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
        X_test, Y_test, Z_test, XZ_test, device)
    return (X_train_, Y_train_, Z_train_, XZ_train_), \
        (X_test_, Y_test_, Z_test_, XZ_test_)


