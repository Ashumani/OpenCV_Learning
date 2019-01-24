import cv2 , time, pandas
from datetime import datetime

video = cv2.VideoCapture(0)


# check, frame = video.read()
# print(check)
# print(frame)
# time.sleep(3)
#
# cv2.imshow("Capture",frame)
# cv2.waitKey(0)
# video.release()
# cv2.destroyAllWindows()

###################################################################

# a = 1
# while True:
#     a = a+1
#     check,frame = video.read()
#     print(frame)
#     gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("capture",gray)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# print(a)
# video.release()
# cv2.destroyAllWindows()

####################################################################

first_frame = None
status_list = [None,None]
times= []
df=pandas.DataFrame(columns=["Start","End"])

video = cv2.VideoCapture(0)

while True:
    check,frame = video.read()
    # print(frame)
    # print(check)
    status=0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    cv2.imshow("capture", gray)
    if first_frame is None:
        first_frame = gray
        continue
delta_frame  = cv2.absdiff(first_frame,gray)
thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
thresh_delta = cv2.dilate(thresh_delta,None, itertations=0)
(_,cnts,_) = cv2.findCounters(thresh_delta.copy(),cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

for contour in cnts:
    if cv2.contourArea(contour) < 1000:
        continue
    status=1
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow('frame', frame)
cv2.imshow('capturing', gray)
cv2.imshow('delta', delta_frame)
cv2.imshow('thresh', thresh_delta)

status_list.append(status)
status_list=status_list[-2:]
if status_list[-1]==1 and status_list[-2]==0:
    times.append(datetime.now())
if status_list[-1]==0 and status_list[-2]==1:
    times.append(datetime.now())

print(status_list)
print(times)
for i in range(0,len(times),2):
    df=df.append({"Start":times,"End":[i+1]},ignore_index=True)
df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
