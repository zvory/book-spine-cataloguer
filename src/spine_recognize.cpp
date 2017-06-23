#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <boost/algorithm/string.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <ctime>
#include <fstream>
#include <vector>
#include "spine_recognize.h"

using namespace cv;
using std::string;
using std::vector;


vector<Mat> detect_text(Mat large){

    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    pyrDown(rgb, rgb);
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours

    vector<Mat> vec {};
    vec.emplace_back(rgb);
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){

        Rect rect = boundingRect(contours[idx]);

        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);

        RotatedRect rrect = minAreaRect(contours[idx]);
        double r = (double)countNonZero(maskROI) / (rrect.size.width * rrect.size.height);


        Scalar color;
        int thickness = 1;
        // assume at least 25% of the area is filled if it contains text
        if (r > 0.6 && 
                (rrect.size.height > 8 && rrect.size.width > 8) // constraints on region size
                // these two conditions alone are not very robust. better to use something 
                //like the number of significant peaks in a horizontal projection as a third condition
           ){
            thickness = 2;
            color = Scalar(0, 255, 0);

            Point2f pts[4];
            rrect.points(pts);

            // matrices we'll use
            Mat M, rotated, cropped;
            // get angle and size from the bounding box
            float angle = rrect.angle;
            Size rect_size = rrect.size;
            // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
            if (rrect.angle < -45.) {
                angle += 90.0;
                swap(rect_size.width, rect_size.height);
            }
            // get the rotation matrix
            M = getRotationMatrix2D(rrect.center, angle, 1.0);
            // perform the affine transformation
            warpAffine(rgb, rotated, M, rgb.size(), INTER_CUBIC);
            // crop the resulting image
            getRectSubPix(rotated, rect_size, rrect.center, cropped); 


            if(cropped.rows >1 && cropped.cols > 1){
                vec.emplace_back(cropped);
                //            namedWindow(std::to_string(vec.size()-1), WINDOW_NORMAL);
                //            imshow(std::to_string(vec.size()-1), cropped );
            }

            for (int i = 0; i < 4; i++) {
                line(rgb, Point((int)pts[i].x, (int)pts[i].y), Point((int)pts[(i+1)%4].x, (int)pts[(i+1)%4].y), color, thickness);
            }
        }
    }

    return vec;
}


int main(int argc,char** argv) {
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    //Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        exit(1);
    }

    Mat image;
    string input_window_name {"Input"}, output_window_name {"Output"};

    // Read in image
    if (argc>=2)
        image = imread( argv[1], IMREAD_COLOR ); // Load an image
    else 
        image = imread("../data/1.png", IMREAD_COLOR);

    if( image.empty() )
        return -1;

    vector<Mat> text = detect_text(image);
    Mat text_highlighted = text[0];

    namedWindow( input_window_name, WINDOW_NORMAL);
    imshow(input_window_name, image);

    for (int i = 0; i < text.size(); ++i) {
        namedWindow(std::to_string(i), WINDOW_NORMAL);
        imshow(std::to_string(i), text[i]);
    }


    vector <Mat> post_adaptive_threshold {};

    for (int i =1; i < text.size(); ++i) {
        post_adaptive_threshold.emplace_back(Mat{});
        cvtColor(text[i], post_adaptive_threshold[i-1], CV_BGR2GRAY );
        adaptiveThreshold(post_adaptive_threshold[i-1], post_adaptive_threshold[i-1], 256,  ADAPTIVE_THRESH_GAUSSIAN_C,  THRESH_BINARY, 3, 0);

        Mat curr = post_adaptive_threshold[i-1];
        api->SetImage((uchar*)curr.data, curr.size().width, curr.size().height, curr.channels(), curr.step1());
        api->Recognize(0);
        
//        api->TesseractExtractResult();
//        tesseract::ResultIterator *ri = api->GetIterator();
//        if (ri!=0) {
//            float conf = ri->Confidence();
//            std::cout<< conf << std::endl;
//        }

                    
        std::cout << boost::trim_copy(string{api->GetUTF8Text()}) << std::endl;

    }


    namedWindow(output_window_name, WINDOW_NORMAL);
    imshow(output_window_name, text_highlighted);

    imwrite( "../output/output" + std::to_string(std::time(0)) + ".jpg", text_highlighted);  

    waitKey(0);
    waitKey(0);
    api->End();

    return 0;
}

