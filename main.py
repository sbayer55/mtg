import cv2
import numpy as np

FLIP_VERTICAL = 0
FLIP_HORIZONTAL = 1

CARD_SIZE_MIN = 10000
CARD_SIZE_MAX = 30000

PINK = (255, 0, 255)
GREEN = (0, 255, 0)

def noop(*args, **kwargs):
    pass

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def run():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    cv2.createTrackbar("Slider A", "preview", 0, 255, noop)
    cv2.createTrackbar("Slider B", "preview", 0, 255, noop)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        slider_a = cv2.getTrackbarPos("Slider A", "preview")
        slider_b = cv2.getTrackbarPos("Slider B", "preview")
        frame = cv2.flip(frame, FLIP_HORIZONTAL)

        frame_raw = frame.copy()

        frame = cv2.GaussianBlur(frame, (7, 7), 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)

        # Could be better if I can resude the noise
        # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # frame_data = cv2.Canny(frame, slider_a, slider_b)
        frame_data = cv2.Canny(frame, 80, 255)

        kernel = np.ones((5, 5))
        frame_data = cv2.dilate(frame_data, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(frame_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.drawContours(frame_raw, contours, -1, PINK, 7)

        cards = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= CARD_SIZE_MIN and area < CARD_SIZE_MAX:
                cnt_len = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                cnt_sides = len(approx)
                hull = cv2.convexHull(cnt)

                if cnt_sides == 4:
                    min_rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(min_rect)
                    box = np.int0(box) # box = [[261 208] [371 190] [395 338] [284 356]]

                    card = four_point_transform(frame_raw, box)
                    card = cv2.flip(card, FLIP_VERTICAL)
                    # card = cv2.resize(card, (630, 880), cv2.INTER_AREA)
                    card = cv2.resize(card, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
                    cards.append(card)

                    cv2.drawContours(frame_raw, cnt, -1, PINK, 7)
                    cv2.drawContours(frame_raw, [box], -1, (0, 0, 255), 7)

                    # x, y, w, h = cv2.boundingRect(approx)
                    # cv2.rectangle(frame_raw, (x, y), (x + w, y + h), GREEN, 5)

                    # cv2.putText(frame_raw, f"Points: {len(approx)}", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, GREEN, 2)
                    # cv2.putText(frame_raw, f"Area: {int(area)}", (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, GREEN, 2)

        # frame_data = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # frame = np.vstack((frame_raw, frame_data))

        if len(cards) > 0:
            all_frames = [frame_raw]
            all_frames.extend(cards)
            frame = np.vstack(np.array(all_frames))

        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")


if __name__ == '__main__':
    run()


