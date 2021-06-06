# Hip-OA-from-Radiographs-MATLAB-
MATLAB code for training a CNN using transfer learning for radiographic hip osteoarthritic features extraction 

This code takes two classes (default) to train a pretained CNN Resnet18 (default) to extract radiogrpahic features of hip osteorathritis. 

(c) Robel Gebre. University of Oulu <br />
2021

# How to run
Step 1: Set the path for datasets and import as imageDatastores <br />
Step 2: Specify the CNN (MATLAB will requesat a download if it doesn't exist) <br />
Step 3: Freeze the top layers (optinal) <br />
Step 4: Peform minor data augmentations randomly in the imageDatastores <br />
Step 5: Speficy the training options (minibatch size, initial learning rate, optimizer) <br />
        There is an option to stop the training if accuracy doesn't improve by setting stop training function in the training options from "inf" to a finite number of iterations. <br />
Step 6: Train the network <br />
Step 7: Run code to end to see results <br />

# Results
Training, validation and test set accuracies, confusion matrices, and occulusive sensitivity feature maps.
