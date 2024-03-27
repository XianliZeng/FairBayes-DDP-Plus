import numpy as np



def measures_from_Yhat_DemPa(Y1hat, Y0hat, Y1, Y0):
    assert isinstance(Y1hat, np.ndarray)
    assert isinstance(Y0hat, np.ndarray)
    assert isinstance(Y1, np.ndarray)
    assert isinstance(Y0, np.ndarray)

    datasize = len(Y1) + len(Y0)
    # Accuracy
    acc = ((Y1hat == Y1).sum() + (Y0hat == Y0).sum()) / datasize
    # Misclassification rate
    MC = 1 - acc

    # DDP
    DDP = np.mean(Y1hat) - np.mean(Y0hat)
    return acc, MC, DDP


def solvet(delta,s1,s2,beta):
    q = (s1 / s2)**(1 / beta)
    D_max = q * (1-np.log(q))
    if delta>=D_max:
        t = 0
    else:
        tmin = 0
        tmax = s1 / 2
        while tmax-tmin>1e-5:
            t = (tmin + tmax) / 2
            qt = ((s1 - (2 * t)) / s2)**(1 / beta)
            D =  qt * (1-np.log(qt))
            if delta >= D:
                tmax = t
            else:
                tmin = t

    return t

def misclssification_rate(t,s1,s2,beta):
    if t >= s1 / 2:
        R = 1 / 2 - s2 / 2 / (beta + 1) ** 2
    else:
        qt = ((s1 - (2 * t)) / s2)**(1 / beta)
        R = 1/2 - s1 * qt / 2 * (1 - np.log(qt)) -s2 / 2 / (beta + 1) * (
            (1 - qt ** (beta+1))/(beta + 1) + qt ** (beta + 1) * np.log(qt))
    return  R



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






