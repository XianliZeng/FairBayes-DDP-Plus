import numpy as np
import pandas as pd

import numpy as np
import traceback
import random




def cal_acc(eta, Y, Z, t1, t0):
    Yhat = np.zeros_like(eta)
    Yhat[Z == 1] = (eta[Z == 1] > t1)
    Yhat[Z == 0] = (eta[Z == 0] > t0)
    acc = (Yhat == Y).mean()
    return acc

def cal_disparity(eta,Z,t1,t0):
    disparity = (eta[Z==1]>t1).mean() - (eta[Z==0]>t0).mean()
    return disparity


def cal_t_with_delta(eta1, eta0,delta,pre_level):
    pa_hat = len(eta1) / (len(eta1) + len(eta0))
    D0 = np.mean(eta1 > 1 / 2 ) - np.mean(eta0 > 1 / 2)
    if delta >= D0:
        t = 0
    else:
        tmin = 0
        tmax = min(pa_hat, (1 - pa_hat))

        while tmax - tmin > pre_level:
            tmid = (tmin + tmax)/2
            t1 = 0.5 + tmid / 2 / pa_hat
            t0 = 0.5 - tmid / 2 / (1 - pa_hat)
            DDP  = np.mean(eta1 > t1) - np.mean(eta0 > t0)
            if DDP > delta:
                tmin = tmid
            else:
                tmax = tmid
        t = (tmin + tmax) / 2
    return t

def threshold_DemPa(eta1, eta0, l1,l0,rn,Deltan, delta=0, pre_level=1e-5):

    pa_hat = len(eta1) / (len(eta1) + len(eta0))
    D1 = np.mean(eta1 > 1 / 2 + l1) - np.mean(eta0 > 1 / 2 - l0 )
    if D1 > delta:
        tildedelta = delta
    else:
        tildedelta = 0
    deltaplus = delta + Deltan
    deltaminus = delta - Deltan
    tmid = cal_t_with_delta(eta1,eta0,delta,pre_level)
    tmin = cal_t_with_delta(eta1,eta0,deltaplus,pre_level)
    tmax = cal_t_with_delta(eta1,eta0,deltaminus,pre_level)

    if tmid - tmin <= rn:
        that = tmin
        flag = 0
    elif tmax - tmid <=rn:
        that = tmax
        flag = 1
    else:
        that = tmid
        flag = 2
    t1hat  = 0.5 + that / 2 / pa_hat
    t0hat = 0.5 - that / 2 / (1 - pa_hat)
    pi1p = np.mean(eta1 > t1hat + l1)
    pi1e = np.mean((eta1 <= t1hat + l1) & (eta1 > t1hat - l1))
    pi0p = np.mean(eta0 > t0hat + l0)
    pi0e = np.mean((eta0 <= t0hat + l0) & (eta0 > t0hat - l0))

    diff1 = pi0p - pi1p + tildedelta
    diff0 = pi1p - pi0p - tildedelta

    if pi1e ==0:
        tau1 = 0
    elif diff1 <=0:
        tau1 = 0
    elif diff1>pi1e:
        tau1 = 1
    else:
        tau1 = diff1/pi1e


    if pi0e == 0:
        tau0 = 0
    elif diff0 <=0:
        tau0 = 0
    elif diff0 > pi0e:
        tau0 = 1
    else:
        tau0 = diff0/pi0e
    return t1hat, t0hat, tau1,tau0,flag




def measures_from_Yhat_DemPa(Y1hat, Y0hat, Y1, Y0):
    assert isinstance(Y1hat, np.ndarray)
    assert isinstance(Y0hat, np.ndarray)
    assert isinstance(Y1, np.ndarray)
    assert isinstance(Y0, np.ndarray)

    datasize = len(Y1) + len(Y0)
    # Accuracy
    acc = ((Y1hat == Y1).sum() + (Y0hat == Y0).sum()) / datasize
    # Misclassification rate

    # DDP
    DDP = np.mean(Y1hat) - np.mean(Y0hat)
    return acc,  DDP




def measures_from_Yhat_Others(Y, Z, Yhat=None, threshold=0.5):
    assert isinstance(Y, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert Yhat is not None
    assert isinstance(Yhat, np.ndarray)

    if Yhat is not None:
        Ytilde = (Yhat >= threshold).astype(np.float32)
    assert Ytilde.shape == Y.shape and Y.shape == Z.shape

    # Accuracy
    acc = (Ytilde == Y).astype(np.float32).mean()
    # DP
    DDP = abs(np.mean(Ytilde[Z == 0]) - np.mean(Ytilde[Z == 1]))
    # EO

    data = [acc, DDP]
    columns = ['acc', 'DDP']
    return pd.DataFrame([data], columns=columns)






def xi(gamma,z,theta):
    if z >= theta:
        return 0
    elif z <= (theta-gamma):
        return theta - z - gamma / 2
    else:
        return (theta - z) ** 2 / 2 / gamma

def gradxi(gamma,tau,mu,f):
    if (tau * mu) >= f:
        return 0
    elif f <= ((tau * mu)-gamma):
        return -1 * tau
    else:
        return -1 * tau * (f - mu * tau) / gamma





def fint_mu(eta,Z,rho,T,alpha):
    gamma = 0.01
    samplesize = len(eta)
    mu = np.array([0.0])
    mulist = [mu]
    for t in range(T):
        index = np.random.choice(samplesize,1)
        tau = (Z[index] - rho).astype(np.float64)
        f = (2 * eta[index] - 1).astype(np.float64)
        Delta = gradxi(gamma,tau,mu,f)
        mu = mu - alpha * Delta
        mulist.append(mu)
    muarray = np.array(mulist)
    return muarray.mean()




def predictor(eta,Z,gamma,mu,rho):
    f = eta * 2 -1
    Yhat = (f - mu)/gamma
    Yhat[f <= (mu * (Z-rho))] = 0
    Yhat[f >= (gamma + mu * (Z-rho))] = 1

    return Yhat






def postprocess(alpha_seed_and_kwargs, postprocessor_factory,
                probas, labels, groups, n_test, n_post,dataset_name):

    if len(alpha_seed_and_kwargs) == 2:
        alpha, seed = alpha_seed_and_kwargs
        kwargs = {}
    else:
        alpha, seed, kwargs = alpha_seed_and_kwargs

  # Split the remaining data into post-processing and test data


    train_probas_post = probas[:n_post]
    train_labels_post = labels[:n_post]
    train_groups_post = groups[:n_post]
    test_probas = probas[n_post:]
    test_labels = labels[n_post:]
    test_groups = groups[n_post:]

    if alpha == np.inf:
    # Evaluate the unprocessed model
        postprocessor = None
        test_preds = test_probas.argmax(axis=1)
    else:
        try:
      # Post-process the predicted probabilities
            postprocessor = postprocessor_factory().fit(train_probas_post,
                                                  train_groups_post,
                                                  alpha=alpha,
                                                  **kwargs)
      # Evaluate the post-processed model
            test_preds = postprocessor.predict(test_probas, test_groups)
        except Exception:
            print(f"Post-processing failed with alpha={alpha} and seed={seed}:\n{traceback.format_exc()}",flush=True)
            data = [seed, dataset_name, alpha, None, None]
            columns = ['seed', 'dataset', 'alpha', 'acc', 'disparity']
            df_test = pd.DataFrame([data], columns=columns)

            return df_test

    acc = (test_preds == test_labels).mean()
    disparity = np.abs((test_preds[test_groups==1]).mean() - (test_preds[test_groups==0]).mean())


    data = [seed,dataset_name,alpha, acc, np.abs(disparity)]
    columns = ['seed','dataset','alpha','acc', 'disparity']

    df_test = pd.DataFrame([data], columns=columns)

    return df_test




def threshold_flipping(pa,eta, Yhat,Y,Z,level):

    s = ((1-Z)/(1-pa) - Z/pa) * (2* Yhat-1) /   ( (2 * eta - 1)*(2 * Yhat-1))
    ssort = s.argsort()
    n1 = Yhat[Z==1].sum()
    n0 = Yhat[Z==0].sum()
    acc = (Yhat == Y).mean()
    acc_max = 0
    tstar = -100000
    n = len(Z)
    p1 = n1/n
    p0= n0/n
    dpstar  = -199
    for idx in ssort:
        t = s[idx]

        acc = acc + (1-2*(Yhat[idx]==Y[idx]))/n
        if Z[idx]==1:
            p1 = p1 + (1- 2 * Yhat[idx])/n
        if Z[idx]==0:
            p0 = p0 + (1- 2 * Yhat[idx])/n
        dp = np.abs(p1 / pa - p0/(1-pa))
        if (dp<=level) & (acc>acc_max):
            tstar = t
            acc_max = acc
            dpstar = dp


    return tstar





def postpreocessing_flipping(pa,eta, Yhat,Z,t):

    s = ((1-Z)/(1-pa) - Z/pa) * (2* Yhat-1) /     ( (2 * eta - 1)*(2 * Yhat-1))
    Yhat_new = (s/t<=1) * Yhat  +    (s/t>1) * (1- Yhat)

    return Yhat_new



def cal_acc_PPF(Yhat, Y):

    acc = (Yhat == Y).mean()
    return acc

def cal_disparity_PPF(Yhat,Z):


    disparity = (Yhat[Z==1]).mean() - (Yhat[Z==0]).mean()

    return disparity
