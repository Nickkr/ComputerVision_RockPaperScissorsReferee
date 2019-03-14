import cv2
from scipy.spatial import distance
import hotword

playerOneScore = 2
playerTwoScore = 2
winnerText = ""
cam = cv2.VideoCapture(0)
cv2.namedWindow("RockPaperScissor - Game")

def checkForWinner():
    if playerOneScore == 3 or playerTwoScore == 3:
        return True
    else:
        return False
def evaluateGameRound(playerOnePose, playerTwoPose):
    if playerOnePose == 'undefined' or playerTwoPose == 'undefined':
        return -1
    if playerOnePose == playerTwoPose:
        return 0
    elif playerOnePose == 'rock' and playerTwoPose == 'scissor':
        return 1
    elif playerOnePose == 'paper' and playerTwoPose == 'rock':
        return 1
    elif playerOnePose == 'scissor' and playerTwoPose == 'paper':
        return 1
    else:
        return 2
def isFingerContracted(fingerArray, centerPoint):
    maxDiff = 0
    for keypoint in fingerArray:
        if keypoint is None:
            pass
        diff = distance.euclidean(list(centerPoint), list(keypoint))
        if diff > maxDiff:
            maxDiff = diff
    distanceFromTip = distance.euclidean(list(centerPoint), list(fingerArray[-1]))
    if distanceFromTip > maxDiff:
        return False
    elif distanceFromTip == maxDiff:
        return False
    else:
        return True
def classifyPose(points):
    centerPoint = points[0]
    print(centerPoint)
    isIndexFingerContracted = isFingerContracted(points[5:9], centerPoint)
    isRingFingerContracted = isFingerContracted(points[13:17], centerPoint)
    isLittleFingerContracted = isFingerContracted(points[17:21], centerPoint)
    isMiddleFingerContracted = isFingerContracted(points[9:13], centerPoint)
    print("IndexFinger " + str(isIndexFingerContracted))
    print("RingFinger " + str(isRingFingerContracted))
    print("LittleFinger " + str(isLittleFingerContracted))
    print("MiddleFinger " + str(isMiddleFingerContracted))
    if (not isIndexFingerContracted) and (not isMiddleFingerContracted) and isRingFingerContracted and isLittleFingerContracted:
        return 'scissor'
    elif (not isIndexFingerContracted) and (not isMiddleFingerContracted) and (not isRingFingerContracted) and (not isLittleFingerContracted):
        return 'paper'
    elif (isIndexFingerContracted + isMiddleFingerContracted + isRingFingerContracted + isLittleFingerContracted) >= 3:
        return 'rock'
    else:
        return 'undefined'
def evaluate(frame):

    BODY_PARTS = {"Wrist": 0,
                  "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                  "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7,
                  "IndexFingerDistal": 8,
                  "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11,
                  "MiddleFingerDistal": 12,
                  "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15,
                  "RingFingerDistal": 16,
                  "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19,
                  "LittleFingerDistal": 20,
                  }

    POSE_PAIRS = [["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                  ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                  ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                  ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                  ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                  ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                  ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                  ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                  ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                  ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]]

    protoTextPath = 'models/hand/pose_deploy.prototxt'
    caffemodelPath = 'models/hand/pose_iter_102000.caffemodel'
    inWidth = 368
    inHeight = 368
    inScale = 0.003922
    thr = 0.05

    net = cv2.dnn.readNet(cv2.samples.findFile(protoTextPath), cv2.samples.findFile(caffemodelPath))

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv2.dnn.blobFromImage(frame, inScale, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    assert (len(BODY_PARTS) <= out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]


        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
            cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 100000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    classifyPose(points)

    return frame, points


while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Speak sentence \"Rock Paper Scissors\" to trigger evaluation", (100, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.putText(frame, "PLAYER 1: " + str(playerOneScore), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.putText(frame, "PLAYER 2: " + str(playerTwoScore), (410, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.putText(frame, winnerText, (190, 250), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
    cv2.line(frame, (325, 60), (325, 419), (0, 0, 0), 2)

    cv2.imshow("RockPaperScissor - Game", frame)
    if not ret:
        break
    hotword.run()
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # Space pressed
        winnerText = ''
        cropFirst = frame[58:420, 0:325]
        cropSecond = frame[58:420, 325:650]

        framePlayerOne, pointsPlayerOne = evaluate(cv2.flip(cropFirst, 1))
        framePlayerTwo, pointsPlayerTwo = evaluate(cropSecond)

        print("Points Player One: " + str(pointsPlayerOne))
        print("Points Player Two: " + str(pointsPlayerTwo))
        playerOnePose = classifyPose(pointsPlayerOne)
        playerTwoPose = classifyPose(pointsPlayerTwo)
        winner = evaluateGameRound(playerOnePose, playerTwoPose)
        if winner == '-1':
            pass
        elif winner == 1:
            playerOneScore += 1
            if checkForWinner():
                winnerText = 'Player 1 has won!'
        elif winner == 2:
            playerTwoScore += 1
            if checkForWinner():
                winnerText = 'Player 2 has won!'
        print("Pose Player 1: " + playerOnePose)
        print("Pose Player 2: " + playerTwoPose)

        cv2.imshow('PlayerOne', framePlayerOne)
        cv2.imshow('PlayerTwo', framePlayerTwo)

cam.release()
cv2.destroyAllWindows()

