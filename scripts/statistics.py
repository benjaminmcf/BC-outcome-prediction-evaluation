import pandas as pd

LABEL='ispos'
DATA = pd.read_csv('../DATA/PROCESSED_DATA/JOINED_PROCESSED.csv')

experiment_cols_crp = [
    '[NE-SFL(ch)]',
    '[NE-FSC(ch)]',
    '[LY-Y(ch)]',
    '[MO-X(ch)]',
    '[MO-Z(ch)]',
    '[NE-WX]',
    '[NE-WY]',
    '[LY-WY]',
    '[LY-WZ]',
    '[MO-WX]',
    '[MO-WY]',
    '[MO-WZ]',
    'HGB(g/L)',
    'RDW-CV(%)',
    'PLT(10^9/L)',
    'WBC(10^9/L)',
    'MONO%(%)',
    'BASO%(%)',
    'EO%(%)',
    'LYMPH%(%)',
    'NEUT%(%)',
    'BASO#(10^9/L)',
    'MONO#(10^9/L)',
    'LYMPH#(10^9/L)',
    'NEUT#(10^9/L)',
    'MCHC(g/L)', 
    'MCV(fL)', 
    'RBC(10^12/L)',
    'EO#(10^9/L)',
    'crp',
    'ispos',
    #'Age',
    'NLR',
    'MLR'

]

columns_of_interest = []
for col in experiment_cols_crp:
    if col == LABEL:
        continue
    else:
        columns_of_interest.append(col)



NEG_MEDIAN_LIST = []
NEG_IQR_LIST = []
POS_MEDIAN_LIST = []
POS_IQR_LIST = []


for cat in [0, 1]:
    data_copy_temp = DATA[DATA[LABEL] == cat]
    print("**************************************** \n")
    print(f"Category: {cat}")
    MEDIAN_LIST = []
    IQR_LIST = []
    if cat == 0:
        print("Negative BC result")
    else:
        print("Positive BC result")
    print()

    for col in columns_of_interest:
        if col == 'crp':    
            data_copy_temp = data_copy_temp[data_copy_temp[col] >= 0.0]
        median = data_copy_temp[col].median().round(3)
        minimum = data_copy_temp[col].min().round(3)
        maximum = data_copy_temp[col].max().round(3)
        q1 = data_copy_temp[col].quantile(0.25).round(3)
        q3 = data_copy_temp[col].quantile(0.75).round(3)
        iqr = (q3 - q1).round(3)

        MEDIAN_LIST.append(median)
        IQR_LIST.append(iqr)

        print(f"Column: {col}")
        print(f"Category: {cat}")
        print("Minimum:", minimum)
        print("Maximum:", maximum)
        print("Median:", median)
        print("Interquartile Range (IQR):", iqr)
        print("Q1:", q1)
        print("Q3:", q3)
        print()
    
    if cat == 0:
        NEG_MEDIAN_LIST = MEDIAN_LIST.copy()
        NEG_IQR_LIST = IQR_LIST.copy()
    else:
        POS_MEDIAN_LIST = MEDIAN_LIST.copy()
        POS_IQR_LIST = IQR_LIST.copy()

neg_length = len(DATA[DATA[LABEL] == 0])
pos_length = len(DATA[DATA[LABEL] == 1])

assert neg_length == 54
assert pos_length == 10

statistics = {'Feature': columns_of_interest,
         f'NEG Median ({neg_length})': NEG_MEDIAN_LIST,
         f'NEG IQR ({neg_length})': NEG_IQR_LIST,
         f'POS Median({pos_length})': POS_MEDIAN_LIST,
         f'POS IQR ({pos_length})': POS_IQR_LIST}
df = pd.DataFrame(data=statistics)
df.to_csv('output_files/statistics.csv', index=False)