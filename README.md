# Landmark-Controlled-Macro

#### Download Zip or Clone the repo, then run the program with this command:
```
python app.py
```

This program uses DLIB's Facial Landmark pre-trained model to detect and estimate 68 facial points.
![image](https://github.com/chowafu/Landmark-Controlled-Macro/assets/73844275/a81a7c47-4661-4510-9f88-d9075389cad5)

Alongside the facia landmark, the program also uses 9-point hand landmark detector to be used as a 'cursor'.
![image](https://github.com/chowafu/Landmark-Controlled-Macro/assets/73844275/c7837bf8-1c12-49b4-ac89-7c5eefb1726b)

Using the detected facial landmarks, the program calculates a new landmark to be used as the macro location.
The macro contains instructions to execute an application, such as Google Chrome or Word Document, upon being triggered.
In this case, the new macro location are the left and right cheeks.
![image](https://github.com/chowafu/Landmark-Controlled-Macro/assets/73844275/2fc5621d-b5ae-48f1-8311-25780195ced3)

The program then detects if the cheek has been touched with the user's index finger.
Upon being touched, the macro runs the application assigned to it.
