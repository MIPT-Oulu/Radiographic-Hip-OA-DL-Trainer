function lgraph = replacinglayersCUSTOM(lgraph,numClasses,l)
%% This code replaces layers from the network
% Description: Takes input from Main_hip_OA_trainer
% % Inputs: layer graph, classes, number of layers to freeze (l)
% (probs), positive class, modelname and title string for the plots.
%
% % Outputs: A structure containing the ROC and PR perfromances
%
% (C) Robel K. Gebre
% Medical Imaging, Physics and Technology (MIPT)
% University of Oulu, Oulu, Finland
% 2021
%%
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
% [learnableLayer,classLayer]; 

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',1920, ...
        'BiasLearnRateFactor',1000);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',1920, ...
        'BiasLearnRateFactor',1000);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:l) = freezeWeights(layers(1:l));
lgraph = createLgraphUsingConnections(layers,connections);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
end

