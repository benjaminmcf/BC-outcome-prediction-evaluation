import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve,precision_recall_curve
from tqdm import tqdm

SAVE_FIGS = True
PERFORM_UNIVARIATE_ANALYSIS = True

# Used for univariate threshold analysis
variable_list =[
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
    #'Age',
    'NLR',
    'MLR']

def univariate_analysis(data=None,variable_list=None,label=None):
     
     VARIABLE_ANALYSIS_LIST = []
     VARIABLE_AUC_LIST = []
     for variable in tqdm(variable_list):
        VARIABLE_ANALYSIS_LIST.append(variable)
        data_copy = data.copy()
        data_copy = data_copy[data_copy[variable] >= 0.0]
        median = data_copy[variable].median()
        minimum = data_copy[variable].min()
        maximum = data_copy[variable].max()
        # print()
        # print(f"Column: {variable}")
        # print("Median:", median)
        # print("Minimum:", minimum)
        # print("Maximum:", maximum)
        data_copy.reset_index(drop=True, inplace=True)
        tpr = []
        fpr = []
        thresholds = []

    
        median_predictions = []
        for i in range(0,len(data_copy)):
            if data_copy.at[i,variable] > median:
                median_predictions.append(1)
            else:
                median_predictions.append(0)

        CM = confusion_matrix(data_copy[label].values,median_predictions)
        # print(f'{CM} -- {median}')


        for thres in np.arange(minimum,maximum,0.05):
            auc_predictions = []
            for i in range(0,len(data_copy)):
                if data_copy.at[i,variable] > thres:
                    auc_predictions.append(1)
                else:
                    auc_predictions.append(0)

            CM = confusion_matrix(data_copy[label].values,auc_predictions)
            
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            tpr.append(TPR)
            fpr.append(FPR)
            thresholds.append(thres)
        
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc,3)

        VARIABLE_AUC_LIST.append(roc_auc)


        # print(f'data length: {len(data_copy)}')
        # print("AUC")
        # print(roc_auc)
        if variable == 'crp':
            fpr_crp = fpr
            tpr_crp = tpr
            roc_auc_crp = roc_auc
            assert roc_auc == 0.606
        if variable == '[NE-WY]':
             fpr_nwy = fpr
             tpr_nwy = tpr 
             roc_auc_nwy = roc_auc
             assert roc_auc == 0.727
        if variable == 'RDW-CV(%)':
             fpr_rdw = fpr
             tpr_rdw = tpr
             roc_auc_rdw = roc_auc
             assert roc_auc == 0.707
        if variable == 'WBC(10^9/L)':
             fpr_wbc = fpr
             tpr_wbc = tpr
             roc_auc_wbc = roc_auc
             assert roc_auc == 0.641
        if variable == 'NEUT%(%)':
             fpr_neut = fpr
             tpr_neut = tpr
             roc_auc_neut = roc_auc
             assert roc_auc == 0.864
        if variable == 'NEUT#(10^9/L)':
             fpr_neutc = fpr
             tpr_neutc = tpr
             roc_auc_neutc = roc_auc
             assert roc_auc == 0.680
        if variable == 'NLR':
             fpr_nlr = fpr
             tpr_nlr = tpr
             roc_auc_nlr = roc_auc
             assert roc_auc == 0.820

     plt.figure(figsize=(8, 6))
     plt.plot(fpr_crp, tpr_crp, color='#e60049', lw=2, label='CRP (auROC = {:.3f})'.format(roc_auc_crp))
     plt.plot(fpr_nwy, tpr_nwy, color='#0bb4ff', lw=2, label='NE-WY (auROC = {:.3f})'.format(roc_auc_nwy))
     plt.plot(fpr_rdw, tpr_rdw, color='#50e991', lw=2, label='RDW-CV(%) (auROC = {:.3f})'.format(roc_auc_rdw))
     plt.plot(fpr_wbc, tpr_wbc, color='#e6d800', lw=2, label='WBC(10^9/L) (auROC = {:.3f})'.format(roc_auc_wbc))
     plt.plot(fpr_neut, tpr_neut, color='#9b19f5', lw=2, label='NEUT%(%) (auROC = {:.3f})'.format(roc_auc_neut))
     plt.plot(fpr_neutc, tpr_neutc, color='#ffa300', lw=2, label='NEUT#(10^9/L) (auROC = {:.3f})'.format(roc_auc_neutc))
     plt.plot(fpr_nlr, tpr_nlr, color='#dc0ab4', lw=2, label='NLR (auROC = {:.3f})'.format(roc_auc_nlr))
     plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random (auROC = 0.5)')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('Receiver operating characteristic curve')
     plt.legend(loc='lower right')
     if SAVE_FIGS:
        plt.savefig(f'output_files/Figure 2.png', dpi=300, bbox_inches='tight')

     univariate_results = {'Feature': VARIABLE_ANALYSIS_LIST,
            f'AUC': VARIABLE_AUC_LIST,
            }
     df = pd.DataFrame(data=univariate_results)
     df.sort_values(by='AUC',ascending=False,inplace=True)
     df.to_csv('output_files/univariate_results.csv', index=False)




def evaluate_model(data=None,model=None,features=None,label=None):
    print('*'*20)
    print(f'Evaluating model: {model}')


    print(data[features].columns)

    X = data[features].copy().values
    y = data[label].copy().values

    features_list_test = features.copy()
    features_list_test.append(label)

    features_list_train = features.copy()
    features_list_train.append(label)

   
    # Add different thresholds to the list to test model performance at different thresholds
    # Current code produces results for >= 0.5
    threshold_list = [0.5]
    for threshold in threshold_list:
        
        y_pred = (model.predict_proba(X)[:,1] >= threshold).astype(bool)
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
        roc_auc = auc(fpr, tpr)
        

        precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1])

        auc_prc = auc(recall, precision)

        print()
        print(f"AUC -> {round(roc_auc,3)}")
        print()
        print(f'AUC-PRC -> {auc_prc}')
        print()
        print(f'Confusion Matrix - threshold {threshold}')
        cm = confusion_matrix(y, y_pred)
        print(cm)
        print()
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        print(f'TN: {TN}')
        print(f'FN: {FN}')
        print(f'TP: {TP}')
        print(f'FP: {FP}')
        print()
        
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        npv = TN / (TN + FN)
        ppv = TP / (TP + FP)

        print(f"Sensitivity: {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"Negative Predictive Value: {npv:.3f}")
        print(f"Positive Predictive Value: {ppv:.3f}")
        print()
        class_names = ['NEG BC', 'POS BC']
        if model.__class__.__name__ == 'XGBClassifier':

            # Add asserts for testing

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'XG model - threshold {threshold}')
            plt.xlabel('Predicted class')
            plt.ylabel('True class')
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
            if SAVE_FIGS:
                plt.savefig(f'output_files/cm_xg_CBC_DIFF_CPD_1.5_boruta - threshold {threshold}.png', dpi=300, bbox_inches='tight')
        
        
        if model.__class__.__name__ == 'RandomForestClassifier':


            # Add asserts for testing

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'RF model - threshold {threshold}')
            plt.xlabel('Predicted class')
            plt.ylabel('True class')
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
            if SAVE_FIGS:
                plt.savefig(f'output_files/cm_rf_CBC_DIFF_1_boruta - threshold {threshold}.png', dpi=300, bbox_inches='tight')
    
   
    return roc_auc, tpr, fpr



def main():

    MODELS_PATH = '../MODELS'
    FEATURES_PATH = '../FEATURES'

    xg_model = pickle.load(open(f'{MODELS_PATH}/xg_CBC_DIFF_CPD_1.5_boruta.sav', 'rb'))
    with open(f'{FEATURES_PATH}/xg_CBC_DIFF_CPD_1.5_boruta.txt') as f:
                xg_feature_list = f.read().splitlines()

    rf_model = pickle.load(open(f'{MODELS_PATH}/rf_CBC_DIFF_1_boruta.sav', 'rb'))
    with open(f'{FEATURES_PATH}/rf_CBC_DIFF_1_boruta.txt') as f:
                rf_feature_list = f.read().splitlines()

    model_list = [xg_model,rf_model]
    feature_list = [xg_feature_list,rf_feature_list]

    data_p1 = pd.read_csv('../DATA/PROCESSED_DATA/P1_PROCESSED.csv')

    unique,counts = np.unique(data_p1['ispos'],return_counts=True)
        

    data_p2 = pd.read_csv('../DATA/PROCESSED_DATA/P2_PROCESSED.csv')
    unique,counts = np.unique(data_p2['ispos'],return_counts=True)


    data_joined = pd.concat([data_p1, data_p2])

    # ispos count for joined data
    unique,counts = np.unique(data_joined['ispos'],return_counts=True)
    print("number of pos and neg BC results after data processing: ",unique,counts)
    print()
    print(data_joined['organism'].value_counts())
    print()

    data_joined.to_csv('../DATA/PROCESSED_DATA/JOINED_PROCESSED.csv',index=False)
    
    
    for t in zip(model_list,feature_list):
        roc_auc,tpr,fpr = evaluate_model(data=data_joined,model=t[0],features=t[1],label='ispos')
        if t[0].__class__.__name__ == 'XGBClassifier':
            fpr_xg = fpr
            tpr_xg = tpr
            roc_auc_xg = roc_auc
        if t[0].__class__.__name__ == 'RandomForestClassifier':
            fpr_rf = fpr
            tpr_rf = tpr
            roc_auc_rf = roc_auc

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='b', lw=2, label='Random forest (auROC = {:.3f})'.format(roc_auc_rf))
    plt.plot(fpr_xg, tpr_xg, color='g', lw=2, label='XGBoost (auROC = {:.3f})'.format(roc_auc_xg))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random (auROC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='lower right')
    if SAVE_FIGS:
        plt.savefig(f'output_files/Figure 1.png', dpi=300, bbox_inches='tight')
    
    
    print()
    if PERFORM_UNIVARIATE_ANALYSIS:       
        univariate_analysis(data=data_joined,variable_list=variable_list,label='ispos')
    


if __name__ == '__main__':
    main()
