%% Detecting hip osteoarthritis on clinical CT: A deep learning application based on 2-D summation images derived from CT
%
% (C) Robel K. Gebre
% Medical Imaging, Physics and Technology (MIPT)
% University of Oulu, Oulu, Finland
% 2021
%

%%%% Description: A Deep Learning Network to Extract Image Features for rHOA Classification
% This code reads a Training, Validation and Test sets from folders ->
% loads an ImageNet trained CNN from MATLAB (download the pretrined model
% at first) -> 
% Then for quicker training time, the weights of the first layers can be
% frozen by specifiying the number ->
% Then Augmentation can be performed on the Training and/or Validation
% depending on the options ("full" vs "resize") specified ->
% The basic training options necessary to train a model are given.
% Additoinal options can be added.

%% Start
clear; close all; clc; 
rng(2020,"multFibonacci"); %Random Seed Number

%% (Optoinal) Set parallel pool and gpu - Uncomment if machine is capable
% delete(gcp('nocreate')); %Stop any running parallel processors
% reset(gpuDevice(1)); %Reset GPU memory
% [availableGPUs,gpuIndx] = gpuDeviceCount("available");
% useGPUs = [1 availableGPUs];
% parpool('local',numel(useGPUs));
% spmd
%     gpuDevice(useGPUs(labindex));
% end
%% Load Folders To Create Training, Validation and Testing Sets

imdsTrain = imageDatastore('A:\ROBEL\GEBRE',...
    'IncludeSubfolders',true, 'LabelSource','foldernames');

fprintf('Training set (%4.0f) : OA vs no OA = %4.0f, %4.0f \n',...
    numel(find(imdsTrain.Labels == "OA")) + ...
    numel(find(imdsTrain.Labels == "noOA")),...
    numel(find(imdsTrain.Labels == "OA")),...
    numel(find(imdsTrain.Labels == "noOA")));
%
imdsValidation = imageDatastore('A:\ROBEL\GEBRE',...
    'IncludeSubfolders',true, 'LabelSource','foldernames');

fprintf('Validation set (%4.0f) : OA vs no OA = %4.0f, %4.0f \n',...
    numel(find(imdsValidation.Labels == "OA")) + ...
    numel(find(imdsValidation.Labels == "noOA")), ...
    numel(find(imdsValidation.Labels == "OA")),...
    numel(find(imdsValidation.Labels == "noOA")));
%
imdsTest = imageDatastore('A:\ROBEL\GEBRE',...
    'IncludeSubfolders',true, 'LabelSource','foldernames');

fprintf('Test set (%4.0f) : OA vs no OA = %4.0f, %4.0f \n',...
    numel(find(imdsTest.Labels == "OA")) + ...
    numel(find(imdsTest.Labels == "noOA")), ...
    numel(find(imdsTest.Labels == "OA")),...
    numel(find(imdsTest.Labels == "noOA")));

%% Load Pretrained Network
net = resnet18;
modelname = 'resnet18';
% analyzeNetwork(net)
net.Layers(1);
inputSize = net.Layers(1).InputSize;
numClasses = numel(categories(imdsTrain.Labels)); %Number of classes to be trained
%% Replace and/or Freeze Layers
if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end
%
layerstofreeze = 5; %first layers
lgraph = replacinglayersCUSTOM(lgraph,numClasses,layerstofreeze); %The final network to be used for training
%% Augmentation (Optional)
[augimdsTrain] = dataAugentationCustom(inputSize,imdsTrain,'full');
[augimdsValidation] = dataAugentationCustom(inputSize,imdsValidation,'full');
[augimdsTest] = dataAugentationCustom(inputSize,imdsTest,'resize');
%
minibatch = preview(augimdsTrain);
imshow(imtile(minibatch.input));   
%% Training Options
miniBatchSize = 32; %32 and 64 most commonly used sizes
valFrequency = floor(augimdsValidation.NumObservations./miniBatchSize);
%
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ... 
    'Shuffle',"every-epoch",...
    'MaxEpochs',10, ... %Change to tune
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'InitialLearnRate',0.0001,...%Change to tune
    'Verbose',false, ...%Change to see numberical output
    'LearnRateSchedule',"piecewise",...
    'Plots','training-progress',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,inf),... %Change inf to stop if accuracy doesn't change
    'ValidationPatience',inf,...
    'ExecutionEnvironment','gpu');

%% Train network
net = trainNetwork(augimdsTrain,lgraph,options);
%% Performance Matrix
% 1. Training 
[YPred_train,probs_train] = classify(net,augimdsTrain);
accuracy_train = mean(YPred_train == imdsTrain.Labels); %Accuracy

pOSclass = 'OA'; %Specify the correct positice class name
[ROCperf_train,PrecRecallPerf_train] = classificationperformance(imdsTrain.Labels,...
    YPred_train,probs_train(:,1),...
    pOSclass, ...
    modelname, ...
    'Training Sample');

% 1. Validation
[YPred_validation,probs_val] = classify(net,augimdsValidation);
accuracy_validation = mean(YPred_validation == imdsValidation.Labels);
pOSclass = 'OA';

[ROCperf_val,PrecRecallPerf_val] = classificationperformance(imdsValidation.Labels,...
    YPred_validation,probs_val(:,1),...
    pOSclass, ...
    modelname, ...
    'Validation Sample'); 
% 2. Test

[YPred_test,probs_test] = classify(net,augimdsTest);
accuracy_test = mean(YPred_test == imdsTest.Labels);
pOSclass = 'OA';

[ROCperf_test,PrecRecallPerf_test] = classificationperformance(imdsTest.Labels,...
    YPred_test,...
    probs_test(:,1),...
    pOSclass, ...
    modelname, ...
    'Separate Test Sample');
%% Occlusion Senstivity Map
a = 1; %change value to go to a specific image. (A for loop can be included here to view all the images)
randomimage = imdsTest.Files{a, 1};
img = imread(randomimage);
% img = imread('Ace.Left_Anon005P.tif');
inputSize = net.Layers(1).InputSize(1:2);
img = imresize(img,inputSize);
classes = net.Layers(end).Classes;

[YPred_maptestim,scores] = classify(net,img);

[~,topIdx] = maxk(scores,3);
topScores = scores(topIdx);
topClasses = classes(topIdx);

figure;
imshow(img)
titleString = compose("%s (%.2f)",topClasses,topScores');
title(sprintf(join(titleString, "; ")));

map = occlusionSensitivity(net,img,YPred_maptestim); %score map
imshow(img,'InitialMagnification', 200)
hold on
imagesc(map,'AlphaData',0.3)
colormap jet
colorbar

title(sprintf("Occlusion sensitivity (%s)", YPred_maptestim));
hold off
% end
%%
delete(gcp('nocreate')) %Stop any running parallel pools
