function [augimdsData] = dataAugentationCustom(inputSize,imdsData,typeCustom)
%% This code perfoms data augmentation on random images of the dataset
% Description: Takes input from Main_hip_OA_trainer
% % Inputs: network inputsize, image datastore, type of augmentation.
%
% % Outputs: augmentated data
%
% (C) Robel K. Gebre
% Medical Imaging, Physics and Technology (MIPT)
% University of Oulu, Oulu, Finland
% 2021
%%
strlengh = length(typeCustom);
if strlengh == 4
    pixelRange = [-5 5];
    scaleRange = [0.75 1.2];
        
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation',pixelRange, ...
        'RandScale',scaleRange,...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    
    augimdsData = augmentedImageDatastore(inputSize,imdsData,"DataAugmentation",imageAugmenter);
    fprintf('Full data-augmentation Done!');
else
    augimdsData = augmentedImageDatastore(inputSize,imdsData);
    fprintf('Simple data-augmentation Done!');
   
end
end

