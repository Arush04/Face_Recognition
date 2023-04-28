import face_recognition as fr
import cv2 as cv
import streamlit as st
from PIL import Image
import os
import numpy as np
import csv

def encode_faces(folder):
    list_people_encoding = []
    for filename in os.listdir(folder):
        known_image = fr.load_image_file(f'{folder}{filename}')
        known_encoding = fr.face_encodings(known_image)[0]
        list_people_encoding.append((known_encoding, filename))
    return list_people_encoding

def create_frame(output_image, location, label):
    top, right, bottom, left = location

    cv.rectangle(output_image, (left, top), (right, bottom), (255,0,0), 2)
    cv.rectangle(output_image, (left, bottom+20), (right, bottom), (255,0,0), cv.FILLED)
    cv.putText(output_image, label, (left+3, bottom+14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255),1)


def find_target_face(uploaded_file):
    target_image = fr.load_image_file(uploaded_file)
    target_array = np.array(target_image)

    output_image = np.copy(target_array)
    
    face_location = fr.face_locations(target_array)

    recognized_faces = []

    for person in encode_faces("your folder location"):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces(encoded_face, target_encoding, tolerance=0.6)
        print(f'{is_target_face} {filename}')
        if face_location:
            face_number = 0
            for location in face_location:
                if face_number < len(is_target_face) and is_target_face[face_number]:
                    label = filename
                    x = label[:label.index(".")]
                    recognized_faces.append(x)
                    create_frame(output_image, location, x)

                face_number += 1

    var1 = set(recognized_faces)
    recognized_faces = list(var1) 
    
    with open('recognized_faces.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for name in recognized_faces:
            writer.writerow([name])

    return output_image, recognized_faces


def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

st.title('Face Recognition App')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

target_image = fr.load_image_file(uploaded_file)
target_encoding = fr.face_encodings(target_image)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    target_image, recognized_faces = find_target_face(uploaded_file)

    st.image(target_image, channels='RGB', caption='Recognized Faces')

    if recognized_faces:
        st.write('The following pre-defined faces were recognized:')
        for name in recognized_faces:
            st.write(name)

        with open('recognized_faces.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Recognized Faces'])
            for name in recognized_faces:
                writer.writerow([name])

        st.write('Results written to recognized_faces.csv')

    else:
        st.write('No pre-defined faces were recognized in the image.')

