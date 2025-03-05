from minio import Minio
import cv2

client = Minio(
    'localhost:9000',
    access_key = 'EftVSRmijXRsItFf2zcb',
    secret_key = 'CMuwmyhht7O766VsbpY68WTbystVhJ43AF6CqQnt',
    secure=False
)

buckets = client.list_buckets()
for obj in buckets:
    print(obj)
test = client.presigned_get_object('test','visual-02-small_1.gif')
cap = cv2.VideoCapture(test)
while cap.isOpened():
    ret, frame = cap.read()
    print(frame)
    if not ret:
        break