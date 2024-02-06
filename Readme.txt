1. Clone this repository

2. Upload the Animal_and_Birds_Detection.ipynb and train.py to Google Drive or Runtime at Google Colab and 
   train.py file shoulb be like this "/content/train.py" or your specific Google Drive Folder

3. Annotate your images using LabelImg tool

4. Make one folder by Naming Dataset, follow the below hierarchy
                                            
                                  		Dataset Folder
			  ____________________________|____________________________
			 |			       				   |
			Train                       				 Test
        (It contains both images and annotations - No subfolder)    (It contains testing images and annotations -No Subfolder)

5. Upload the Dataset folder to Your Google Drive and mount the google drive to Google Colab

6. Open the train.py file and pass the train and test dataset paths respectively