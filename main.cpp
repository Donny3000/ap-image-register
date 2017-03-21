#include <iostream>
#include <stdio.h>
#include "apImageRegister.hpp"

using namespace std;
using namespace cv;
using namespace ap;

/**
 * @function main
 */
int main(int argc, char** argv)
{
  double minVal, maxVal;
  Point minLoc, maxLoc;

  /*TODO: 
   * > Bash script: DICOM -> JPG or make a C++ class to manage this.
   * > C++ Library or class to manage graphics.
   * > GUI to explore images to make the program easier to use.
   * > Args parser tool for bash or perl scripts
   */
  //ImageRegister imgRegister("ct.jpg","pet.jpg");
  cout << "Template Image: " << argv[1] << endl;
  cout << "Moving Image  : " << argv[2] << endl;

  ImageRegister imgRegister(argv[1], argv[2]);

  // Crop out information row in midwave image
  Mat midwave_img = imgRegister.getFixedImage();
  Mat image1_crop = midwave_img( cv::Rect(0, 1, midwave_img.cols, midwave_img.rows-1) );

  // Remove dead pixels from midwave image
  Mat image1_median;
  minMaxLoc(image1_crop, &minVal, &maxVal);
  Mat dead_px_mask = (image1_crop == maxVal);
  medianBlur(image1_crop, image1_median, 3);
  image1_median.copyTo(image1_crop, dead_px_mask);

  imgRegister.setFixedImage( image1_crop );
  
  /* Histograms calculation */
  Mat hist_fixed = imgRegister.calHistogram( imgRegister.getFixedImage() );
  Mat hist_moving = imgRegister.calHistogram( imgRegister.getMovingImage() );
  
  /* Joint Histogram calculation */
  
  Mat joint_hist = imgRegister.calJointHistogram(imgRegister.getFixedImage(), imgRegister.getMovingImage());
  
  minMaxLoc(joint_hist, &minVal, &maxVal, &minLoc, &maxLoc);
  
  cout << "Joint Histogram minimal value : " << minVal << endl;
  cout << "Joint Histogram maximal value : " << maxVal << endl;
  
  /* Draw histograms */
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound((double)hist_w/imgRegister.getHistSize()); 
  
  Mat histImageFixed( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0));
  Mat histImageMoving( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
  
  normalize(hist_fixed, hist_fixed, 0, histImageFixed.rows, NORM_MINMAX, -1, Mat() );
  normalize(hist_moving, hist_moving, 0, histImageMoving.rows, NORM_MINMAX, -1, Mat() );

  for( int i = 1; i < imgRegister.getHistSize(); i++ )
  {
      line( histImageFixed, Point( bin_w*(i-1), hist_h - cvRound(hist_fixed.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist_fixed.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImageMoving, Point( bin_w*(i-1), hist_h - cvRound(hist_moving.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist_moving.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );      
  }
  /* Testing */
  
  cout << "Template image entropy : " << imgRegister.calEntropy(imgRegister.getFixedImage()) << endl;
  cout << "Moving image entropy   : " << imgRegister.calEntropy(imgRegister.getMovingImage()) << endl;
  cout << "Joint Entropy          : " << imgRegister.calJointEntropy(imgRegister.getFixedImage(),imgRegister.getMovingImage()) << endl;
  cout << "Mutual Information     : " << imgRegister.calMutualInformation(imgRegister.getFixedImage(),imgRegister.getMovingImage()) << endl;
  
  imgRegister.calMaxMutualInformationValue(imgRegister.getFixedImage(), imgRegister.getMovingImage(), 1, 1);
  
  /* Display images and histograms */
  
  namedWindow("Histogram: Template image", CV_WINDOW_AUTOSIZE );
  imshow("Histogram: Template image", histImageFixed );

  namedWindow("Template image", CV_WINDOW_AUTOSIZE );
  imshow("Template image", imgRegister.getFixedImage() );

  namedWindow("Histogram: Moving image", CV_WINDOW_AUTOSIZE );
  imshow("Histogram: Moving image", histImageMoving );

  namedWindow("Moving image", CV_WINDOW_AUTOSIZE );
  imshow("Moving image", imgRegister.getMovingImage() );

  namedWindow("Joint Histogram", CV_WINDOW_AUTOSIZE );
  imshow("Joint Histogram", joint_hist );
   
  waitKey(0);  
  destroyAllWindows();

  return ImageRegister::OK;
}
