# For Video and Image Conversion
import cv2 as cv

# For Hand Landmark Detection
import mediapipe as mp

# For Face Landmark Detection
import dlib

# For browser opening using open() method
import webbrowser as browse

# To avoid triggering the open() method multiple times
# in a few milliseconds using the sleep() method,
# since our program is inside a continuous loop.
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Use webcam input
vid_stream = cv.VideoCapture(0)


def landmark_detector():

    with mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:

        while vid_stream.isOpened():

            success, image = vid_stream.read()

            if not success:
                print("Skipping Frame")
                continue

            image = cv.flip(image, 1)
            image4dlib = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image4mp = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            imgHeight, imgWidth, _ = image.shape

            faces = detector(image4dlib)
            results = hands.process(image4mp)

            # The program triggers when we initialize indexTip_x and indexTip_y to n <= 5.
            # because if no face or hand is detected, every variable
            # remains initialized to zero and lines 130 & 134 returns true.
            # So this variable is set to any number other than n <= 5.
            indexTip_x = indexTip_y = 20
            # Initialize all variables to zero
            r_mid_x = r_mid_y = l_mid_x = l_mid_y = 0
            right_x_dist = right_y_dist = 0
            left_x_dist = left_y_dist = 0

            for face in faces:
                landmarks = predictor(image4dlib, face)

                for point in range(0, 68):
                    x = landmarks.part(point).x
                    y = landmarks.part(point).y

                    cv.circle(image, (x, y), 2, (255, 255, 0), -1)

                # Calculate Cheek Landmark by finding the middle point of
                # both Left and Right Nose and Outer Face Landmark.
                # x_mid_point = (x1 + x2) // 2
                # y_mid_point = (y1 + y2) // 2
                r_mid_x = (landmarks.part(14).x + landmarks.part(35).x) // 2
                r_mid_y = (landmarks.part(11).y + landmarks.part(45).y) // 2
                l_mid_x = (landmarks.part(2).x + landmarks.part(31).x) // 2
                l_mid_y = (landmarks.part(5).y + landmarks.part(36).y) // 2

                # Draw a point at the calculated Cheek Landmark
                cv.circle(image, (r_mid_x, r_mid_y), 2, (0, 0, 255), -1)
                cv.circle(image, (l_mid_x, l_mid_y), 2, (0, 0, 255), -1)

                # Put a label at the calculated Cheek Landmark
                cv.putText(
                    image,
                    "Github",
                    (r_mid_x - 30, r_mid_y + 10),
                    cv.FONT_HERSHEY_PLAIN,
                    1,
                    [255, 255, 255],
                    1,
                    cv.LINE_4,
                )
                cv.putText(
                    image,
                    "Facebook",
                    (l_mid_x - 40, l_mid_y + 10),
                    cv.FONT_HERSHEY_PLAIN,
                    1,
                    [255, 255, 255],
                    1,
                    cv.LINE_4,
                )

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Get coordinates of Index Finger Tip (IFT)
                    indexTipLandmark = hand_landmarks.landmark[8]

                    # Convert IFT's Coordinates to Pixel Coordinate (X, Y)
                    indexTipCoord = mp_drawing._normalized_to_pixel_coordinates(
                        indexTipLandmark.x, indexTipLandmark.y, imgWidth, imgHeight
                    )

                    # index 0 = x-value
                    indexTip_x = indexTipCoord[0]
                    # index 1 = y-value
                    indexTip_y = indexTipCoord[1]

            # Get the distance of the IFT to the Left and Right Cheek.
            right_x_dist = abs(indexTip_x - r_mid_x)
            right_y_dist = abs(indexTip_y - r_mid_y)
            left_x_dist = abs(indexTip_x - l_mid_x)
            left_y_dist = abs(indexTip_y - l_mid_y)

            # Check if IFT is less than 5 pixels away from Right Cheek
            if right_x_dist <= 5 and right_y_dist <= 5:
                browse.open("https://chowafu.github.io/Portfolio/index.html")
                time.sleep(0.5)
            # Check if IFT is less than 5 pixels away from Left Cheek
            elif left_x_dist <= 5 and left_y_dist <= 5:
                browse.open("https://www.facebook.com/")
                time.sleep(0.5)
            else:
                print("Not Touching Cheeks.")

            cv.imshow("Detect", image)

            key = cv.waitKey(1)
            if key == 27:
                break

        vid_stream.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    landmark_detector()
