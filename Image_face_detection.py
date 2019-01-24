import cv2

# img = cv2.imread('C:\\Users\\HP\\PycharmProjects\\OpenCV_Learning\\faces.jpg',1)

# # print(img)
# print(type(img))
# print(img.shape)
#
# cv2.imshow("Legend",img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

##############################################################
# Resize the image
# resized = cv2.resize(img,(600,600)) # resize excel with pixel
#resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))  #Resize img with division opertor
# resized = cv2.resize(img, (int(img.shape[1]*2),int(img.shape[0]*2)))   #Resize img with Multiplication Oerator
#
# cv2.imshow("legend",resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
# create a CascadeClassifier Object
face_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\Anaconda3\\pkgs\\libopencv-3.4.1-h875b8b8_3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')


# Reading the image as it is
img = cv2.imread('C:\\Users\\HP\\PycharmProjects\\OpenCV_Learning\\faces.jpg',1)

#Reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Search the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,
                                      minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)
# resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0])))

cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()