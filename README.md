# TMA_classification

## Project Description

Digital pathology has the potential to help pathologists and improve the way diagnoses are made. In this project, we used patch-based Convolutional Neural Networks (CNN) to classify Tissue Microarrays (TMA) benign or tumor. This method processes large, high-resolution data efficiently, making diagnoses faster and supporting pathologists in their decisions. Our experiments showed that the EfficientNetV2-S model achieved 93% accuracy on the test set. In the future, we plan to work with larger datasets and use Explainable AI to make the model's predictions easier to understand and more useful in clinical settings.

## Code Explaination
1. **tma_clss.sh**:  
   - This script uses the SLURM scheduler to run `main.py` on the TALC server with GPU resources.  
   - The job is set up with 1 GPU, 16GB of memory, and 2 CPU cores.  
   - `main.py` is executed with the JOB_ID as an argument to easily manage output files and results.  

2. **reorganize_folder.py**:  
   - This script preprocesses the dataset by filtering and organizing raw images and masks into a format suitable for `main.py`.  
   - The process has two main steps:  
     1. **Filtering by Black Pixel Ratio**: It selects images where the black pixel ratio is below 70%, ensuring clean input data.  
     2. **Classification Using Masks**: Images are categorized into 'tumor' and 'benign' directories based on the white pixel ratio in the masks (threshold: 10%).  
   - This preprocessing is done for both training and validation datasets.  

3. **main.py**:  
   - This script trains and evaluates a model using the preprocessed data.  
   - It uses `EfficientNet` for binary classification and tracks metrics like Precision, Recall, and F1 Score. It also visualizes results such as ROC curves and confusion matrices.  
   - Early Stopping is applied to avoid overfitting during training.  
   - The trained model and evaluation results are saved for future use.
   
