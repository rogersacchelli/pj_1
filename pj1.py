import numpy as np
import cv2
import argparse

__autor__ = "Roger S. Sacchelli - roger.sacchelli@gmail.com"

__doc__ = """
    -------------------------------------------
    ----- Project 1 | Self-driving Car ND -----
    -----     Predicting Lane Lines       -----
    -------------------------------------------
    """


def main():
    print __doc__

    args = parser()

    # Opening video stream
    for f in args.file.split(','):

        cap = cv2.VideoCapture(f)

        while (cap.isOpened()):

            ret, frame = cap.read()

            # Define a kernel size and apply Gaussian smoothing
            blur_stream = gaussian(kernel_size=5, img=frame)

            # Canny Edge detection
            stream_edges = canny(blur_stream)

            # Next we'll create a masked edges image using cv2.fillPoly()
            mask = np.zeros_like(stream_edges)
            ignore_mask_color = 255

            # This time we are defining a four sided polygon to mask
            vertices = polyg_4_vertices(imshape=frame.shape)

            # Applying polygon
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            masked_edges = cv2.bitwise_and(stream_edges, mask)

            lines, line_image = hough_trans(frame, masked_edges)

            # Iterate over the output "lines" and draw lines on a blank image
            if lines is not None:

                lane_lines = lanes_drawing(lines, frame.shape)

                try:
                    for l in lane_lines:
                        all_lines = np.concatenate((all_lines, l))
                except Exception as e:
                    all_lines = np.concatenate(lane_lines)
                    print e

                average_lines = smoothed_line(all_lines[::-1])

                # This method directly prints every point to stream
                for line in average_lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

                # Create a "color" binary image to combine with line image
                # color_edges = np.dstack((stream_edges, stream_edges, stream_edges))

                # Draw the lines on the edge image
                lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

                cv2.imshow('frame', lines_edges)

            elif lines is None:
                cv2.imshow('frame', frame)

            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def parser():
    parser = argparse.ArgumentParser(description='Challenge-4')
    parser.add_argument("-f", "--file", type=str, help='video stream to process')
    parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity")
    args = parser.parse_args()
    return args


def gaussian(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(blur_stream):
    low_threshold = 100
    high_threshold = 250
    return cv2.Canny(blur_stream, low_threshold, high_threshold)


def polyg_4_vertices(imshape):

    return np.array([[(imshape[1] / 10, imshape[0]), (imshape[1] / 2.5, imshape[0] / 1.5),
                      (imshape[1] / 1.6, imshape[0] / 1.5), (imshape[1] / 1.1, imshape[0])]],
                    dtype=np.int32)

    # return np.array([[(0, imshape[0] / 1), (0, 0),
    #                   (imshape[1], 0), (imshape[1], imshape[0] / 1)]],
    #                 dtype=np.int32)


def hough_trans(frame, masked_edges):
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 3  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments

    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    return cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap), line_image


def adaptative_hough_parameters():
    pass


def lanes_drawing(hough_lines, img_shape):

    """"Receives as inputs the hough transform 'lines'
        Evaluates if each input line can be regarded as lane line

        Determines both angular and linear coefficients for lane line by
        averaging the filtered hough lines

        Extrapolate a lane line by drawing a longer line using previously
        calculated coefficients"""

    # left/right lane line criteria
    x_center = img_shape[1] / 2
    y_center = img_shape[0] / 2

    # separates left and right lane points
    # discard the hough line if it's angular coefficient (m_line)
    # do not pose as a lane angle (left lane < -0.3 | right lane > 0.3)

    for l in hough_lines:
        # check which portion of image line l strictly belongs to
        try:
            m_line = float(l[0][3] - l[0][1]) / float(l[0][2] - l[0][0])
            if (l[0][0] and l[0][2]) <= x_center and (l[0][1] and l[0][3]) >= y_center*1.1:
                if m_line < -0.3:
                    try:
                        left_band = np.concatenate((left_band, l[:]))
                    except:
                        left_band = np.concatenate(([l[:]]))
            elif (l[0][0] and l[0][2]) >= x_center and (l[0][1] and l[0][3]) >= y_center*1.1:
                if m_line > 0.3:
                    try:
                        right_band = np.concatenate((right_band, l[:]))
                    except:
                        right_band = np.concatenate(([l[:]]))
        except:
            pass

    x_min_left_points = np.sort(left_band[:, [0]], axis=0)
    y_min_left_points = np.sort(left_band[:, [1]], axis=0)
    x_max_left_points = np.sort(left_band[:, [2]], axis=0)
    y_max_left_points = np.sort(left_band[:, [3]], axis=0)

    x_min_right_points = np.sort(right_band[:, [0]], axis=0)
    y_min_right_points = np.sort(right_band[:, [1]], axis=0)
    x_max_right_points = np.sort(right_band[:, [2]], axis=0)
    y_max_right_points = np.sort(right_band[:, [3]], axis=0)

    x_min_left = int(np.mean(x_min_left_points))
    y_min_left = int(np.mean(y_min_left_points[::-1]))

    x_max_left = int(np.mean(x_max_left_points))
    y_max_left = int(np.mean(y_max_left_points))

    x_min_right = int(np.mean(x_min_right_points))
    y_min_right = int(np.mean(y_min_right_points))

    x_max_right = int(np.mean(x_max_right_points))
    y_max_right = int(np.mean(y_max_right_points[::-1]))

    # extrapolate over line averaging

    m_left = float(y_max_left - y_min_left) / float(x_max_left - x_min_left)

    m_right = float(y_max_right - y_min_right) / float(x_max_right - x_min_right)

    b_left = float(y_max_left) - (m_left * x_max_left)
    b_right = float(y_max_right) - (m_right * x_max_right)

    x_min_left = int(-b_left / m_left) - img_shape[1]
    y_min_left = int(m_left * x_min_left + b_left)

    x_min_right = int(-b_right / m_right) + img_shape[1]
    y_min_right = int(m_right * x_min_right + b_right)

    x_max_left = int(x_center * 0.88)
    y_max_left = int(m_left * x_max_left + b_left)

    x_max_right = int(x_center * 1.12)
    y_max_right = int(m_right * x_max_right + b_right)

    left_lane = np.array([[x_min_left, y_min_left, x_max_left, y_max_left]])
    right_lane = np.array([[x_min_right, y_min_right, x_max_right, y_max_right]])

    return np.array([left_lane, right_lane])


def smoothed_line(lines, mov_avg=10):
    if (lines.shape[0]) / 2 < mov_avg:
        mov_avg = lines.shape[0] / 2

    x_left_min_smooth = int(np.mean(lines[0:len(lines):2, 0][0:mov_avg]))
    y_left_min_smooth = int(np.mean(lines[0:len(lines):2, 1][0:mov_avg]))

    x_left_man_smooth = int(np.mean(lines[0:len(lines):2, 2][0:mov_avg]))
    y_left_man_smooth = int(np.mean(lines[0:len(lines):2, 3][0:mov_avg]))

    x_right_min_smooth = int(np.mean(lines[1:len(lines):2, 0][0:mov_avg]))
    y_right_min_smooth = int(np.mean(lines[1:len(lines):2, 1][0:mov_avg]))

    x_right_man_smooth = int(np.mean(lines[1:len(lines):2, 2][0:mov_avg]))
    y_right_man_smooth = int(np.mean(lines[1:len(lines):2, 3][0:mov_avg]))

    return np.array([[[x_left_min_smooth, y_left_min_smooth, x_left_man_smooth, y_left_man_smooth],
                      [x_right_min_smooth, y_right_min_smooth, x_right_man_smooth, y_right_man_smooth]]])


if __name__ == '__main__':
    main()
