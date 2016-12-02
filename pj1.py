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
    cap = cv2.VideoCapture(args.file)

    while (cap.isOpened()):
        ret, frame = cap.read()

        stream = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_stream = cv2.GaussianBlur(stream, (kernel_size, kernel_size), 0)

        # Canny Edge detection
        low_threshold = 100
        high_threshold = 250
        stream_edges = cv2.Canny(blur_stream, low_threshold, high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(stream_edges)
        ignore_mask_color = 255

        # This time we are defining a four sided polygon to mask
        imshape = stream.shape

        vertices = np.array([[(imshape[1]/10, imshape[0]/1), (imshape[1]/2.5, imshape[0]/1.5),
                              (imshape[1]/1.6, imshape[0]/1.5), (imshape[1]/1.1, imshape[0]/1)]], dtype=np.int32)

        # vertices = np.array([[(imshape[1]/10, imshape[0]), (imshape[1]/2.5, imshape[0]/1.7),
        #                     (imshape[1]/1.8, imshape[0]/1.7), (imshape[1]/1.11, imshape[0])]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(stream_edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180     # angular resolution in radians of the Hough grid
        threshold = 10         # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10    # minimum number of pixels making up a line
        max_line_gap = 6       # maximum gap in pixels between connectable line segments

        line_image = np.copy(frame) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)


        # Iterate over the output "lines" and draw lines on a blank image

        if lines is not None:

            print 'lines: ', len(lines)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 5)

            # Create a "color" binary image to combine with line image
            # color_edges = np.dstack((stream_edges, stream_edges, stream_edges))

            # Draw the lines on the edge image
            lines_edges = cv2.addWeighted(stream, 0.8, line_image, 1, 0)

            cv2.imshow('frame', lines_edges)

        else:
            cv2.imshow('frame',  stream)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parser():
    parser = argparse.ArgumentParser(description='Challenge-4')
    parser.add_argument("-f", "--file", type=str, help='video stream to process')
    parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity")
    args = parser.parse_args()
    return args

def adaptative_hough_parameters():
    pass

if __name__ == '__main__':
    main()