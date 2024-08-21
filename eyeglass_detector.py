import dlib
import cv2
import numpy as np


#==============================================================================
#   1.fungsi konversi format landmark
# Masukan: landmark dalam format dlib
# Output: landmark dalam format numpy
#=============================================================================   
def landmarks_to_np(landmarks, dtype="int"):
    # Dapatkan jumlah landmark
    num = landmarks.num_parts
    
    # menginisialisasi daftar koordinat (x, y).
    coords = np.zeros((num, 2), dtype=dtype)
    
    # lewati 68 landmark wajah dan konversikan
    # ke 2-tuple dari (x, y)-koordinat
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # mengembalikan daftar (x, y)-koordinat
    return coords

#==============================================================================
#   2.Gambar garis regresi & temukan fungsi pupil
# Input: landmark dalam format gambar yang numpy
# Keluaran: koordinat pupil kiri & koordinat pupil kanan
#==============================================================================   
def get_centers(img, landmarks):
    # regresi linier
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) #menarik garis regresi
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

#==============================================================================
#   3.Fungsi Perataan Wajah
# Input: gambar & koordinat pupil kiri & koordinat pupil kanan
# Keluaran: Gambar wajah sejajar
#============================================================================== 
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# alis
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# Jarak interpupil
    scale = desired_dist / dist # rasio skala
    angle = np.degrees(np.arctan2(dy,dx)) # Sudut rotasi
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# Hitung matriks rotasi

    # perbarui komponen terjemahan dari matriks
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

#==============================================================================
#   4.Apakah akan memakai fungsi diskriminan kacamata
# Input: Gambar wajah yang disejajarkan
# Keluaran: nilai diskriminan (true/false)
#============================================================================== 
def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) #gaussian blur

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #deteksi tepi sobel arah-y
    sobel_y = cv2.convertScaleAbs(sobel_y) #Ubah kembali ke tipe uint8
    cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y #matriks kekuatan tepi
    
    #Binarisasi Otsu
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #Menghitung panjang fitur
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] #ekstrak ROI
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#Hitung nilai evaluasi
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#Hitung nilai evaluasi
    measure = measure_1*0.3 + measure_2*0.7
    
    cv2.imshow('roi_1',roi_1)
    cv2.imshow('roi_2',roi_2)
    print(measure)
    
    #Tentukan nilai diskriminan berdasarkan hubungan antara nilai evaluasi dan ambang batas
    if measure > 0.15:#Ambang batas dapat disesuaikan, diuji sekitar 0,15
        judge = True
    else:
        judge = False
    print(judge)
    return judge

#==============================================================================
#   **************************entri fungsi utama***********************************
#==============================================================================

predictor_path = "./data/shape_predictor_5_face_landmarks.dat"#Jalur data pelatihan titik kunci wajah
detector = dlib.get_frontal_face_detector()#detektor pendeteksi wajah
predictor = dlib.shape_predictor(predictor_path)#Prediktor detektor titik kunci wajah

cap = cv2.VideoCapture(0)#Nyalakan kameranya

while(cap.isOpened()):
    #baca bingkai video
    _, img = cap.read()
    
    #ubah menjadi skala abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    rects = detector(gray, 1)
    
    # Operasikan pada setiap wajah yang terdeteksi
    for i, rect in enumerate(rects):
        # mendapatkan koordinat
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face
        
        # Gambar batas dan tambahkan label teks
        cv2.rectangle(img, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Deteksi dan beri label landmark        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # regresi linier
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        
        # keselarasan wajah
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)
        
        # Tentukan apakah akan memakai kacamata
        judge = judge_eyeglass(aligned_face)
        if judge == True:
            cv2.putText(img, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # menunjukkan hasil
    cv2.imshow("Result", img)
    
    k = cv2.waitKey(5) & 0xFF
    if k==27:   #Tekan "Esc" untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
