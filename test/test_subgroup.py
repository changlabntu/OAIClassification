import pandas as pd
import numpy as np
import os
from utils.metrics_classification import GetAUC
from sklearn.metrics import confusion_matrix
import scipy.stats
metrics = GetAUC()

def MOAKS_get_vars(categories, ver):
    moaks_summary = pd.read_excel(os.path.join(os.path.expanduser('~'), 'Dropbox',
                                               'TheSource/OAIDataBase/OAI_Labels/MOAKS/KMRI_SQ_MOAKS_variables_summary.xlsx'))
    moaks_variables = moaks_summary.loc[moaks_summary['CATEGORY'].isin(categories), 'VARIABLE']
    l = list(moaks_variables.values)
    return [x.replace('$$', ver) for x in l]


var_bml = MOAKS_get_vars(['BML Size'], ver='$$')
var_eff = MOAKS_get_vars(['Whole Knee Effusion'], ver='$$')
var_car = MOAKS_get_vars(['Cartilage Morphology'], ver='$$')



x = pd.read_csv('temp.csv')
xbml = x.loc[~x[var_bml[0]].isna(), :] #!!!!


x_painful = x.loc[(x['V$$WOMKP#'] > 0)].reset_index()
ID_has_moaks = x.loc[~x['READPRJ'].isna()]['ID'].unique()  # 587 out of 1446
ID_has_eff = x.loc[~x['V$$MEFFWK'].isna()]['ID'].unique()  # 416 out of 1446
ID_no_eff = [y for y in x['ID'].unique() if y not in ID_has_eff]  # 1030 out of 1446
#pmindexid = x_painful.loc[x_painful['ID'].isin(ID_has_moaks)].index.values
xpm = x_painful.loc[x_painful['ID'].isin(ID_has_eff)]

# demographics
x00 = x.loc[x['VER'] == 0]
xin = x00.loc[x00['ID'].isin(ID_has_eff)]
xout = x00.loc[~x00['ID'].isin(ID_has_eff)]

xout = x00

#xin = x00
print(xin['P01BMI'].mean(), xin['P01BMI'].std(), xout['P01BMI'].mean(), xout['P01BMI'].std())
print(xin['V00AGE'].mean(), xin['V00AGE'].std(), xout['V00AGE'].mean(), xout['V00AGE'].std())
print((xin['P02SEX'] == 2).mean(), (xin['P02SEX'] == 2).std(),
      (xout['P02SEX'] == 2).mean(), (xout['P02SEX'] == 2).std())

scipy.stats.ttest_ind(xin['V00AGE'], xout['V00AGE'])

has_reading = ~xpm['V$$MEFFWK'].isna()
xpmm = xpm.loc[has_reading, :]
xpmm['bml'] = (xpmm.loc[:, var_bml] > 0).sum(1)
xpmm['eff'] = (xpmm.loc[:, var_eff])
xpmm['car'] = (xpmm.loc[:, var_car] > 0).sum(1)


epoch = 100
o_sig = np.load('out/sig_'+ str(epoch) + '.npy')
o_m = np.load('out/m_'+ str(epoch) + '.npy')
l = np.load('label2.npy')
o_sigm = o_sig[has_reading, :]
o_mm = o_m[has_reading, :]
lm = l[has_reading]


bml_tibia = ['V$$MBMSSS', 'V$$MBMSTMA', 'V$$MBMSTLA', 'V$$MBMSTMC', 'V$$MBMSTLC', 'V$$MBMSTMP', 'V$$MBMSTLP']

#condition = xpmm['bml'] > 0
condition = xpmm['car'] > 0
#condition = xpmm['eff'] > 0
#condition = (xpmm.loc[:, bml_tibia].sum(1)) > 0
#condition = xpmm['V$$MSYIC'] > 0
#condition = xpmm['V$$MBMSTMP'] > 0
print('Positive:')
print((metrics(lm[condition], o_mm[condition]), metrics(lm[condition], o_sigm[condition])))

tn, fp, fn, tp = confusion_matrix(lm[condition], (o_mm[condition, 0] < o_mm[condition, 1])).ravel()
specificity_m = tn / (tn+fp)
sensitivity_m = tp / (tp+fn)
tn, fp, fn, tp = confusion_matrix(lm[condition], (o_sigm[condition, 0] < o_sigm[condition, 1])).ravel()
specificity_sig = tn / (tn+fp)
sensitivity_sig = tp / (tp+fn)
print('Sensitivity:', sensitivity_m, sensitivity_sig)
print('Specificity:', specificity_m, specificity_sig)

print('Negative:')
print((metrics(lm[~condition], o_mm[~condition]), metrics(lm[~condition], o_sigm[~condition])))
