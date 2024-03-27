import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP




def print_table1():
    dataset = 'AdultCensus'
    result_FBDP = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FBDP')

    FBDP_acc_mean_list = [result_FBDP[t::50]['acc'].mean() for t in range(50)][::5][:8]
    FBDP_acc_std_list = [result_FBDP[t::50]['acc'].std() for t in range(50)][::5][:8]
    FBDP_ddp_mean_list = [result_FBDP[t::50]['disparity'].mean() for t in range(50)][::5][:8]
    FBDP_ddp_std_list = [result_FBDP[t::50]['disparity'].std() for t in range(50)][::5][:8]
    delta_list = [0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14]
    for l in range(8):
        acc_mean = Decimal(FBDP_acc_mean_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        acc_std = Decimal(FBDP_acc_std_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        ddp_mean = Decimal(FBDP_ddp_mean_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        ddp_std = Decimal(FBDP_ddp_std_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        if l == 7:
            print(f'{delta_list[l]} & {ddp_mean} ({ddp_std}) & {acc_mean} ({acc_std})', end='\\\\\\hline\n')
        else:
            print(f'{delta_list[l]} & {ddp_mean} ({ddp_std}) & {acc_mean} ({acc_std})', end='\\\\\n')

def print_table4():
    dataset = 'AdultCensus'
    result_KDE = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_KDE')
    result_PPOT = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPOT')
    result_PPF = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPF')
    result_ADV = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_ADV')
    result_FBDP = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FBDP')

    KDE_acc_mean = Decimal(result_KDE[9::10]['acc'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    KDE_acc_std = Decimal(result_KDE[9::10]['acc'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    KDE_ddp_mean = Decimal(result_KDE[9::10]['disparity'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    KDE_ddp_std = Decimal(result_KDE[9::10]['disparity'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    ADV_acc_mean = Decimal(result_ADV[10::11]['acc'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    ADV_acc_std = Decimal(result_ADV[10::11]['acc'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    ADV_ddp_mean = Decimal(result_ADV[10::11]['disparity'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    ADV_ddp_std = Decimal(result_ADV[10::11]['disparity'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

    PPOT_acc_mean = Decimal(result_PPOT[::10]['acc'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPOT_acc_std = Decimal(result_PPOT[::10]['acc'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPOT_ddp_mean = Decimal(result_PPOT[::10]['disparity'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPOT_ddp_std = Decimal(result_PPOT[::10]['disparity'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

    PPF_acc_mean = Decimal(result_PPF[::10]['acc'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPF_acc_std = Decimal(result_PPF[::10]['acc'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPF_ddp_mean = Decimal(result_PPF[::10]['disparity'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    PPF_ddp_std = Decimal(result_PPF[::10]['disparity'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

    FBDP_acc_mean = Decimal(result_FBDP[::50]['acc'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    FBDP_acc_std = Decimal(result_FBDP[::50]['acc'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    FBDP_ddp_mean = Decimal(result_FBDP[::50]['disparity'].mean()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    FBDP_ddp_std = Decimal(result_FBDP[::50]['disparity'].std()).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

    print(f'FairBayes-DDP+ (Proposed) &  $\delta=0$ &  {FBDP_acc_mean} ({FBDP_acc_std})  & {FBDP_ddp_mean} ({FBDP_ddp_std})', end='\\\\\\hline\n')
    print(f'ADV &  $\\alpha=5$ &  {ADV_acc_mean} ({ADV_acc_std})  & {ADV_ddp_mean} ({ADV_ddp_std})', end='\\\\\n')
    print(f'KDE &  $\lambda=0.95$ &  {KDE_acc_mean} ({KDE_acc_std})  & {KDE_ddp_mean} ({KDE_ddp_std})', end='\\\\\n')
    print(f'PPOT &  $\delta=0$ &  {PPOT_acc_mean} ({PPOT_acc_std})  & {PPOT_ddp_mean} ({PPOT_ddp_std})', end='\\\\\n')
    print(f'PPF &  $\delta=0$ &  {PPF_acc_mean} ({PPF_acc_std})  & {PPF_ddp_mean} ({PPF_ddp_std})', end='\\\\\\hline\n')


def drow_plot():
    dataset = 'AdultCensus'
    result_KDE = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_KDE')
    result_PPOT = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPOT')
    result_PPF = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPF')
    result_ADV = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_ADV')
    result_FBDP = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FBDP')

    KDE_acc = [result_KDE[t::10]['acc'].mean() for t in range(10)]
    KDE_ddp = [result_KDE[t::10]['disparity'].mean() for t in range(10)]
    ADV_acc = [result_ADV[t::11]['acc'].mean() for t in range(11)]
    ADV_ddp = [result_ADV[t::11]['disparity'].mean() for t in range(11)]
    PPOT_acc = [result_PPOT[t::10]['acc'].mean() for t in range(10)]
    PPOT_ddp = [result_PPOT[t::10]['disparity'].mean() for t in range(10)]
    PPF_acc = [result_PPF[t::10]['acc'].mean() for t in range(10)]
    PPF_ddp = [result_PPF[t::10]['disparity'].mean() for t in range(10)]
    FBDP_acc = [result_FBDP[t::50]['acc'].mean() for t in range(50)]
    FBDP_ddp = [result_FBDP[t::50]['disparity'].mean() for t in range(50)]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(FBDP_acc[::5], FBDP_ddp[::5], marker='*', s=400, label=r'FairBayes-DDP+')
    plt.scatter(ADV_acc, ADV_ddp, marker='>', s=250, label=r'ADV')
    plt.scatter(KDE_acc, KDE_ddp, marker='o', s=250, label=r'KDE')
    plt.scatter(PPOT_acc, PPOT_ddp, marker='s', s=250, label=r'PPOT')
    plt.scatter(PPF_acc, PPF_ddp, marker='^', s=250, label=r'PPF')
    # plt.scatter(Domain_s_acc, Domain_s_ddp, marker='<', color='y', s=400, label=r'Domain')

    plt.xlabel('Accuracy', fontsize=20)
    plt.ylabel('DDP', fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{dataset}')