from deepface import DeepFace

def recognise(frame):
    dfs = DeepFace.find(img_path = frame, db_path = "Module-4/known_faces",model_name="SFace",detector_backend="dlib",distance_metric="euclidean")
    names=[]
    for df in dfs:  
        if df.shape[0]>0:
            names.append(df.iloc[0].identity[21:-4])
    print(names)
    return names


