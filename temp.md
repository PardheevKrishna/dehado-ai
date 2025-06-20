i have a dataset in this format
DEHADO-AI_TRAINING_DATASET
- IMAGES_750
- LABELS_750

images_750 have images for forms, amnd labels have labels in json format like this 
[
    {
        "Field name": "candidatename",
        "Field value": "Dayita Bakshi",
        "Coordinate": [
            914,
            384,
            1550,
            485
        ]
    },
    {
        "Field name": "Father/husbandname",
        "Field value": "Laksh Bakshi",
        "Coordinate": [
            949,
            484,
            1617,
            583
        ]
    },
    {
        "Field name": "Dateofbirth",
        "Field value": "12/27/1975",
        "Coordinate": [
            969,
            586,
            1513,
            677
        ]
    },
]

crop the handwritten images and train on that, ia m working on kaggle, 



 {'WER': 11.92319932109897, 'CER': 5.809792278934005, 'Field Acc': 75.63333333333333, 'Doc Acc': 35.945945945945944, 'Final': 80.53034483188034} - train2
 2025-05-14 10:39:20,707 INFO: Validation Metrics Epoch 30: {'WER': 6.808191272649297, 'CER': 2.7172072691451263, 'Field Acc': 86.696080095028, 'Doc Acc': 86.696080095028, 'Final': 92.67493453888035}








 data cleaning:
 cleaning extra spaces beween words
 fixing incorrect labels mnaually


 why current score on test data
 tell some reason
 other reason can be because 
    of data cleaning - model getting confusd between '-' and '/' because of inconsistent labels in 'dates' and '-' before zipcode of 'address' fields
    dates missing '0'
    aadhar number wrongly labeled
    completely differnt labels for the image
    incorrect languages order in knownlanguages field
most mislabelling is in phase1 of the trainig data

compare other models on unseen data against yours
show error % of model validation vs human validation