// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

// Lab - 01

// Lab - 02
void l2ex124()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
		Mat blue = Mat(height, width, CV_8UC1);
		Mat green = Mat(height, width, CV_8UC1);
		Mat red = Mat(height, width, CV_8UC1);
		Mat hHSV = Mat(height, width, CV_8UC1);
		Mat sHSV = Mat(height, width, CV_8UC1);
		Mat vHSV = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
				blue.at<uchar>(i, j) = b;
				green.at<uchar>(i, j) = g;
				red.at<uchar>(i, j) = r;
				// HSV Transformations
				float bHSV = (float) b / 255;
				float gHSV = (float) g / 255;
				float rHSV = (float) r / 255;
				float maxim = max(bHSV, max(gHSV, rHSV));
				float minim = min(bHSV, max(gHSV, rHSV));
				float c = maxim - minim;
				float h, s, v;
				v = maxim;
				if (v != 0) s = c / v;
				else s = 0;
				if (c != 0) 
				{
					if (maxim == rHSV) h = 60 * (gHSV - bHSV) / c;
					if (maxim == gHSV) h = 120 + 60 * (bHSV - rHSV) / c;
					if (maxim == bHSV) h = 240 + 60 * (rHSV - gHSV) / c;
				}
				else {
					h = 0;
				}
				if (h < 0) h = h + 360;
				hHSV.at<uchar>(i, j) = h / 260 * 255;
				sHSV.at<uchar>(i, j) = s * 255;
				vHSV.at<uchar>(i, j) = v * 255;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		imshow("blue image", blue);
		imshow("green image", green);
		imshow("red image", red);
		imshow("H image", hHSV);
		imshow("S image", sHSV);
		imshow("V image", vHSV);
		waitKey();
	}
}

void l2ex3() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int rows = src.rows;
		int cols = src.cols;
		std::cout << "Prag ";
		int prag;
		std::cin >> prag;
		Mat dst = Mat(rows, cols, CV_8UC1);
		for (int i = 0; i < rows; i++) 
		{
			for (int j = 0; j < cols; j++) 
			{
				uchar val = src.at<uchar>(i, j);
				dst.at<uchar>(i, j) = (val < prag) ? 0 : 255;
			}
		}
		imshow("source image", src);
		imshow("destination image", dst);
		waitKey();
	}
}

bool isInside(Mat img, int i, int j) {
	if ((i < img.rows) && (j < img.cols) && (i >= 0) && (j >= 0))
		return true;
	else
		return false;
}

void l2ex5() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int rows = src.rows;
		int cols = src.cols;
		
		int row = 0, col = 0;
		while (row != -10000 || col != -10000) 
		{
			std::cout << "Point (row, col): ";
			std::cin >> row >> col;
			std::cout << (isInside(src, row, col) ? "Is inside" : "Is NOT inside") << std::endl;
		}
		waitKey();
	}
}

// Lab - 03

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void showHistogram(const std::string& name, double* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	double max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void l3ex1234() {
	int hist[256] = {};
	double fdp[256] = {};
	int histM[256] = {};
	char fname[MAX_PATH];
	int m = 0;
	while (openFileDlg(fname)) {
		std::cout << "m = ";
		std::cin >> m;
		m %= 256;
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				hist[img.at<uchar>(i, j)] ++;
			}
		}
		for (int i = 0; i < 256; i++) {
			fdp[i] = (float) hist[i] / (img.rows * img.cols);
		}
		for (int i = 0; i < 256; i ++) {
			int newBin = (int) (i / 256.0 * m);
			histM[newBin] += hist[i];
		}
		imshow("Image", img);
		showHistogram("Histogram", hist, 256, 256);
		showHistogram("Histogram M", histM, m, 256);
		showHistogram("Histogram FDP", fdp, 256, 256);
		waitKey(0);
	}
}


void l3ex56() {
	int hist[256] = {};
	double fdp[256] = {};
	int histReduced[256] = {};
	int histDithering[256] = {};
	int maxims[256] = {};
	int count = 0;
	int WH = 5;
	double v;
	double TH = 0.0003;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		Mat reduced = imread(fname, IMREAD_GRAYSCALE);
		Mat dithering = imread(fname, IMREAD_GRAYSCALE);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				hist[img.at<uchar>(i, j)] ++;
			}
		}
		for (int i = 0; i < 256; i++) {
			fdp[i] = (double) hist[i] / (img.rows * img.cols);
		}
		for (int i = WH; i <= 256 - WH; i ++) {
			v = 0;
			for (int j = i - WH; j < i + WH; j++) {
				v += fdp[j];
				if (fdp[j] > fdp[i])
					v += 1000;
			}
			v /= (2 * WH + 1);
			if (fdp[i] > v + TH) {
				maxims[++count] = i;
			}
		}
		maxims[0] = 0;
		maxims[++count] = 255;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				int val = img.at<uchar>(i, j);
				int min = 256;
				int close;
				for (int z = 0; z <= count; z++) {
					if (abs(maxims[z] - val) < min) {
						min = abs(maxims[z] - val);
						close = maxims[z];
					}
				}
				reduced.at<uchar>(i, j) = dithering.at<uchar>(i, j) = close;
				histReduced[reduced.at<uchar>(i, j)] ++;
				int error = val - close;
				if (isInside(dithering, i, j + 1)) {
					int corrected = dithering.at<uchar>(i, j + 1) + 7 * error / 16.0;
					dithering.at<uchar>(i, j + 1) = max(0, min(255, corrected));
				}
				if (isInside(dithering, i + 1, j - 1) == true) {
					int corrected = dithering.at<uchar>(i + 1, j - 1) + 3 * error / 16.;
					dithering.at<uchar>(i + 1, j - 1) = max(0, min(255, corrected));
				}
				if (isInside(dithering, i + 1, j) == true) {
					int corrected = dithering.at<uchar>(i + 1, j) + 5 * error / 16.;
					dithering.at<uchar>(i + 1, j) = max(0, min(255, corrected));
				}
				if (isInside(dithering, i + 1, j + 1) == true) {
					int corrected = dithering.at<uchar>(i + 1, j + 1) + error / 16.;
					dithering.at<uchar>(i + 1, j + 1) = max(0, min(255, corrected));
				}
			}
		}
		for (int i = 0; i < dithering.rows; i++) {
			for (int j = 0; j < dithering.cols; j++) {
				histDithering[dithering.at<uchar>(i, j)] ++;
			}
		}
		imshow("Image", img);
		showHistogram("Histogram", hist, 256, 256);
		imshow("Reduced", reduced);
		showHistogram("Histogram - reduced", histReduced, 256, 256);
		imshow("Dithering", dithering);
		showHistogram("Histogram - ditheing", histDithering, 256, 256);
		waitKey(0);
	}
}

// Lab - 04

void makeBW(Mat* src, Vec3b color)
{
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if (src->at<Vec3b>(i, j) == color) {
				src->at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else {
				src->at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}
}

int area(Mat src) {
	int a = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				a++;
			}
		}
	}
	return a;
}

int rSum(Mat src) 
{
	int r = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				r += i;
			}
		}
	}
	return r;
}

int cSum(Mat src)
{
	int c = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				c += j;
			}
		}
	}
	return c;
}

double colorPerimeter(Mat* src, Mat imgBW) {
	double p = 0;
	for (int i = 0; i < imgBW.rows; i++) {
		for (int j = 0; j < imgBW.cols; j++) {
			if ((isInside(imgBW, i, j) && imgBW.at<Vec3b>(i, j) == Vec3b(0, 0, 0))) {
				if ((isInside(imgBW, i + 1, j) && imgBW.at<Vec3b>(i + 1, j) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i + 1, j + 1) && imgBW.at<Vec3b>(i + 1, j + 1) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i + 1, j - 1) && imgBW.at<Vec3b>(i + 1, j - 1) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i - 1, j) && imgBW.at<Vec3b>(i - 1, j) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i - 1, j + 1) && imgBW.at<Vec3b>(i - 1, j + 1) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i - 1, j - 1) && imgBW.at<Vec3b>(i - 1, j - 1) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i, j + 1) && imgBW.at<Vec3b>(i, j + 1) == Vec3b(255, 255, 255)) ||
					(isInside(imgBW, i, j - 1) && imgBW.at<Vec3b>(i, j - 1) == Vec3b(255, 255, 255))) {
					src->at<Vec3b>(i, j) = Vec3b(255, 255, 0);
					p += 1;
				}
			}
		}
	}
	return p * (PI / 4);
}

void colorCenter(Mat* src, int ri, int ci) {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			if (isInside(*src, ri - i, ci)) {
				src->at<Vec3b>(ri - i, ci) = Vec3b(255, 255, 0);
			}
			if (isInside(*src, ri + i, ci)) {
				src->at<Vec3b>(ri + i, ci) = Vec3b(255, 255, 0);
			}
			if (isInside(*src, ri, ci + j)) {
				src->at<Vec3b>(ri, ci + j) = Vec3b(255, 255, 0);
			}
			if (isInside(*src, ri, ci - j)) {
				src->at<Vec3b>(ri, ci - j) = Vec3b(255, 255, 0);
			}

		}
	}
}

double phi(Mat src, int ri, int ci) {
	double num = 0;
	double numi = 0, numi0 = 0, numi1 = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				num += (i - ri) * (i - ri) * (j - ci) * (j - ci);
				numi0 += (j - ci) * (j - ci);
				numi1 += (i - ri) * (i - ri);
			}
		}
	}
	num *= 2;
	numi = numi0 - numi1;
	double ph = atan2(num, numi);
	ph /= 2;
	return ph;
}

void minMaxRC(Mat src, int* rmin, int* rmax, int* cmin, int* cmax) {
	*rmin = INT64_MAX;
	*rmax = INT64_MIN;
	*cmin = INT64_MAX;
	*cmax = INT64_MIN;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				if (i > *rmax) {
					*rmax = i;
				}
				if (i < *rmin) {
					*rmin = i;
				}
				if (j > *cmax) {
					*cmax = j;
				}
				if (j < *cmin) {
					*cmin = j;
				}
			}
		}
	}
}

void histObj(Mat src) {
	int histV[1000] = {}, histO[1000] = {};

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.rows; j++) {
			if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				histO[i]++;
				histV[j]++;
			}
		}
	}
	Mat imgHist(256, 256, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_histV = 0;
	int max_histO = 256;
	for (int i = 0; i < 256; i++) {
		if (histV[i] > max_histV)
			max_histV = histV[i];
		if (histO[i] > max_histO)
			max_histO = histO[i];
	}

	double scaleV = 1.0;
	scaleV = (double)256 / max_histV;
	double scaleO = 1.0;
	scaleO = (double)256 / max_histO;
	int baselineV = 256 - 1, baselineO = 256 - 1;

	for (int x = 0; x < 256; x++) {
		Point p1 = Point(x, baselineO);
		Point p2 = Point(x, baselineO - cvRound(histO[x] * scaleO));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	for (int x = 0; x < 256; x++) {
		Point p1 = Point(baselineV, x);
		Point p2 = Point(baselineV - cvRound(histV[x] * scaleV), x);
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow("Histogram", imgHist);
}

void mouseCallback(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN) {
		Mat src = *((Mat*)param);
		if (src.at<Vec3b>(y, x) == Vec3b(255, 255, 255)) return;
		Mat imgBW = src.clone();
		Mat result = src.clone();
		Mat hist = Mat(src.rows, src.cols, CV_8UC3);
		makeBW(&imgBW, src.at<Vec3b>(y, x));
		int ai = area(imgBW);
		double ri = rSum(imgBW) * 1.0 / ai;
		double ci = cSum(imgBW) * 1.0 / ai;
		colorCenter(&result, (int)ri, (int)ci);
		double p = colorPerimeter(&result, imgBW);
		double ph = phi(imgBW, ri, ci);
		line(result, Point(ci, ri), Point(ci + 30 * cos(ph), ri + 30 * sin(ph)), CV_RGB(0, 255, 255));
		std::cout << "Area = " << ai << "; C(" << ci << ", " << ri << ") ; Perimeter = " << p << " ; Phi = " << ph / (PI * 180) << " ; Subtiere = " << 4 * PI * (ai / p * p) << " ; Compactitate = " << (p > 0) ? 1 / (4 * PI * (ai / p * p)) : 0;
		int rmin, rmax, cmin, cmax;
		minMaxRC(imgBW, &rmin, &rmax, &cmin, &cmax);
		std::cout << " ; R = " << ((cmax - cmin) + 1) * 1.0 / (rmax - rmin + 1) << " ; ";
		histObj(imgBW);
		imshow("Result", result);
	}
}



void l4() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		imshow("Source image", src);
		setMouseCallback("Source image", mouseCallback, &src);
		waitKey();
		destroyAllWindows();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1  - Open image\n");
		printf(" 2  - Open BMP images from folder\n");
		printf(" 3  - Image negative - diblook style\n");
		printf(" 4  - BGR->HSV\n");
		printf(" 5  - Resize image\n");
		printf(" 6  - Canny edge detection\n");
		printf(" 7  - Edges in a video sequence\n");
		printf(" 8  - Snap frame from live video\n");
		printf(" 9  - Mouse callback demo\n");
		printf(" 10 - L1 - \n");
		printf(" 11 - L1 -\n");
		printf(" 12 - L1 -\n");
		printf(" 13 - L1 -\n");
		printf(" 14 - L2 - 1, 2, 4\n");
		printf(" 15 - L2 - 3\n");
		printf(" 16 - L2 - 5\n");
		printf(" 17 - L3 - 1, 2, 3, 4\n");
		printf(" 18 - L3 - 5, 6\n");
		printf(" 19 - L4\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				break;
			case 11:
				break;
			case 12:
				break;
			case 13:
				break;
			case 14:
				l2ex124();
				break;
			case 15:
				l2ex3();
				break;
			case 16:
				l2ex5();
				break;
			case 17: 
				l3ex1234();
				break;
			case 18:
				l3ex56();
				break;
			case 19:
				l4();
				break;
		}
	}
	while (op!=0);
	return 0;
}