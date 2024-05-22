from django.http import HttpResponse, JsonResponse
import json
import pickle
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from sklearn.tree import DecisionTreeClassifier
import requests
from scipy.stats import mode

def Hello(request):
    return HttpResponse("Hello")

@csrf_exempt
def Dimension(request):
    # Extracting data from the request
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    answer = body['answer']
    # Initialize list to store dimension sums
    dim = [10, 10, 15, 10, 10]
    dim_text = ['External Environtment', 'Internal Environtment', 'Behavioral Aspect', 'Cognitive Aspects', 'Indonesia Entrepreneurial Framework']

    # Initialize list to store dimension sums
    hasil_penjumlahan = []

    # Summing each dimension for each row based on dim array
    for row in answer:
        start_index = 0
        row_sum = []
        for dimensi in dim:
            end_index = start_index + dimensi
            row_sum.append(sum(row[start_index:end_index]))
            start_index = end_index
        hasil_penjumlahan.append(row_sum)

    # Convert hasil_penjumlahan into a 2D array
    array_hasil = np.array(hasil_penjumlahan)

    print(hasil_penjumlahan)
    # Convert hasil_penjumlahan into a 2D array
    jumlah_baris_df = len(answer)
    jumlah_dimensi = len(dim)
    array_hasil = np.array(hasil_penjumlahan).reshape((jumlah_baris_df, jumlah_dimensi))

    # Load the classifier from the saved file
    with open("./model/svm_model_dimension.pkl", "rb") as file:
        classifier = pickle.load(file)
        
    # Classify the review using the classifier
    classification = classifier.predict(array_hasil)
    print(f"Predicted classification: {classification}")

    # Ensure classification is a NumPy array
    classification = np.array(classification)

    # Flatten the classification array to 1D
    flattened_classification = classification.flatten()


    # Find the most frequent value in the classifications
    most_frequent_value = mode(flattened_classification)
 

    # Return the result as an HTTP response
    return HttpResponse(int(most_frequent_value.mode[0]))


@csrf_exempt
def Subdimension(request):
    # Extracting data from the request
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    answer = body['answer']
    # Initialize list to store dimension sums
    dim = [5,5,5,5,5,5,5,5,5,5,5]
    indicator_text = ['Goverment support and regulation','Network','Resources','Ability to Innovate Based on Strength ans Specialization','Opportunities Engagement','Entrepreneureship Behavioral Capacity','Entrepreneurial Engagement Capacity','Entrepreneurial Attitude Capacity','Entrepreneurial Capacity','Indonesian Entrepreneurial Activities','Indonesia Individual Attributes']

    # Initialize list to store dimension sums
    hasil_penjumlahan = []

    # Summing each dimension for each row based on dim array
    for row in answer:
        start_index = 0
        row_sum = []
        for dimensi in dim:
            end_index = start_index + dimensi
            row_sum.append(sum(row[start_index:end_index]))
            start_index = end_index
        hasil_penjumlahan.append(row_sum)

    # Convert hasil_penjumlahan into a 2D array
    array_hasil = np.array(hasil_penjumlahan)

    print(hasil_penjumlahan)
    # Convert hasil_penjumlahan into a 2D array
    jumlah_baris_df = len(answer)
    jumlah_dimensi = len(dim)
    array_hasil = np.array(hasil_penjumlahan).reshape((jumlah_baris_df, jumlah_dimensi))

    # Load the classifier from the saved file
    with open("./model/svm_model_subdimension.pkl", "rb") as file:
        classifier = pickle.load(file)
        
    # Classify the review using the classifier
    classification = classifier.predict(array_hasil)
    print(f"Predicted classification: {classification}")

    # Ensure classification is a NumPy array
    classification = np.array(classification)

    # Flatten the classification array to 1D
    flattened_classification = classification.flatten()


    # Find the most frequent value in the classifications
    most_frequent_value = mode(flattened_classification)
 

    # Return the result as an HTTP response
    return HttpResponse(int(most_frequent_value.mode[0]))

