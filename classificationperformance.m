function [ROCperf,PrecRecallPerf] = classificationperformance(labels,predictions,probs,positiveclass,modelname,titlestr)
%% This code reads caclulates the Confusion Matrix, ROC and PR AUCs 
% Description: Takes input from Main_hip_OA_trainer
% % Inputs: True labels (labels), predictions, prediction probabilities
% (probs), positive class, modelname and title string for the plots.
%
% % Outputs: A structure containing the ROC and PR perfromances
%
% (C) Robel K. Gebre
% Medical Imaging, Physics and Technology (MIPT)
% University of Oulu, Oulu, Finland
% 2021
%%
ROCperf = struct();
PrecRecallPerf = struct();
%% 1.1 Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(labels,predictions);
cm.Title = 'Confusion Matrix';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
%% 1.2 AUCROC curve 
[X,Y,T,AUC] = perfcurve(labels,probs,positiveclass);
figure;
plot(X,Y); 
xlabel('False positive rate') 
ylabel('True positive rate')
legend(strcat(modelname,'(AUC = ', num2str(round(AUC,2)),')'),'Location',"southeast")
title(strcat(titlestr,', ROC Curve'))

ROCperf.fpr = X;
ROCperf.tpr = Y;
ROCperf.threshold = T;
ROCperf.ROCAUC = AUC;
%% 1.3 Precision-recall curve 
[Xpr,Ypr,Tpr,AUCpr] = perfcurve(labels,probs,positiveclass,'xCrit', 'reca', 'yCrit', 'prec');
figure;
plot(Xpr,Ypr)
xlabel('Recall') 
ylabel('Precision')
legend(strcat(modelname,'(AUC = ', num2str(round(AUCpr,2)),')'),'Location',"northeast")
title(strcat(titlestr,', Precision-Recall Curve'))

Precision = nanmean(Ypr,'all');
Recall = nanmean(Xpr,'all');
F1_score = 2 * (Precision*Recall/(Precision+Recall));

PrecRecallPerf.Recall = Xpr;
PrecRecallPerf.Precision = Ypr;
PrecRecallPerf.threshold = Tpr;
PrecRecallPerf.PRAUC = AUCpr;
PrecRecallPerf.Precision = Precision;
PrecRecallPerf.Recall = Recall;
PrecRecallPerf.FScore = F1_score;
end

