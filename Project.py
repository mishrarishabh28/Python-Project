import cv2
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam stream
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

         # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with hand detection
        results = hands.process(frame_rgb)

        # Check if a hand is detected
        if results.multi_hand_landmarks:
            # Iterate over detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()