/**
* Based on the https://github.com/opencv/opencv/blob/master/samples/cpp/videocapture_starter.cpp
* A label recognition experimental utility
* 
* Pedro Martins 24/01/2018
*/

//video input
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

//object detection
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>     // std::cout, std::fixed
#include <stdio.h>  
#include <iomanip>      // std::setprecision
#include <fstream>

//denoise
#include <opencv2/photo.hpp>
#include <opencv2/videostab.hpp>


using namespace cv;
using namespace std;

/// Histograms
MatND hist_base_1;
MatND hist_base_2;
MatND hist_base_3;
MatND hist_base_4;
MatND hist_base_5;
MatND hist_test_sample;

void loadHistBase();
void loadHistTestSample();
int compareHistograms();
vector <string>  rawTextDataVec;
vector <string>  loadRawText();
void detectText(Mat frame, char * filename);
vector <vector <string> > itens;
void createSearchPattern();
int findWordInText(int itemPosition);
bool findStringIC(const std::string & strHaystack, const std::string & strNeedle);
int textBasedPrediction();
double avgVec(const vector<double> & v);
void detectAndDisplay(Mat frame, int frameCount);

String r1_cascade_name, r2_cascade_name, r3_cascade_name, r4_cascade_name, r5_cascade_name;
CascadeClassifier r1_cascade;
CascadeClassifier r2_cascade;
CascadeClassifier r3_cascade;
CascadeClassifier r4_cascade;
CascadeClassifier r5_cascade;
String window_name = "Capture - rotule detection";
String window_text = "Tesseract - Text detection";

bool detectR1, detectR2, detectR3, detectR4, detectR5, reset, cascade,
text, t1, t2, t3, t4, t5, h1, h2, h3, h4, h5;
int frameCount;
vector<double> meanR1;
vector<double> meanR2;
vector<double> meanR3;
vector<double> meanR4;
vector<double> meanR5;

//variables to save Mean
double R1, R2, R3, R4, R5;

void init() {

	detectR1 = detectR2 = detectR3 = detectR4 = detectR5 = cascade = true;
	reset = text = t2 = t1 = t3 = t4 = t5 = h1 = h2 = h3 = h4 = h5 = false;
	// face_cascade_name = "bin/data/haar/haarcascade_frontalface_alt.xml";
	// eyes_cascade_name = "bin/data/haar/haarcascade_mcs_mouth.xml";

	r1_cascade_name = "bin/data/rotulos/cascade1.xml";
	r2_cascade_name = "bin/data/rotulos/cascade2.xml";
	r3_cascade_name = "bin/data/rotulos/cascade3.xml";
	r4_cascade_name = "bin/data/rotulos/cascade4.xml";
	r5_cascade_name = "bin/data/rotulos/cascade5.xml";


	//-- 1. Load the cascades for object detection
	if (!r1_cascade.load(r1_cascade_name)) {
		cout << "--(!)Error loading r1\n";
		exit(0);
	};
	if (!r2_cascade.load(r2_cascade_name)) {
		cout << "--(!)Error loading r2\n";
		exit(0);
	};
	if (!r3_cascade.load(r3_cascade_name)) {
		cout << "--(!)Error loading r3\n";
		exit(0);
	};
	if (!r4_cascade.load(r4_cascade_name)) {
		cout << "--(!)Error loading r4\n";
		exit(0);
	};
	if (!r5_cascade.load(r5_cascade_name)) {
		cout << "--(!)Error loading r5\n";
		exit(0);
	};
}

void help(char** av) {
	cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
		<< "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
		<< "q,Q,esc -- quit" << endl
		<< "space   -- save frame" << endl << endl
		<< "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
		<< "\texample: " << " involucros.exe 0, in this case we are assigning device 0." << endl
		<< "\tYou may also pass a video file instead of a device number" << endl
		<< "\texample: " << "involucros.exe" << " video.avi" << endl
		<< "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
		<< "\texample: " << "involucros.exe" << " right%%02d.jpg" << endl;
}

int process(VideoCapture& capture) {
	//int n = 0;
	frameCount = 0;
	char filename[200];
	cout << "press space to save a picture. q or esc to quit" << endl;
	namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;		
	namedWindow(window_text, WINDOW_NORMAL); //resizable window;
	resizeWindow(window_text, 640, 240);
	Mat frame;
	Mat denoised(640, 480, CV_8UC3, Scalar(0, 0, 0));
	Mat textImage(640, 480, CV_8UC3, Scalar(0, 0, 0));

	for (;;) {
		capture >> frame;
		if (frame.empty())
			break;

		char key = (char)waitKey(1); //delay N millis, usually long enough to display and capture input

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
		case ' ': //Save an image
			textImage = frame.clone();
			fastNlMeansDenoisingColored(textImage, textImage, 3, 3, 7, 21); //denoise
			cv::hconcat(frame, textImage, denoised); // horizontal
			detectText(textImage, filename);
			rawTextDataVec = loadRawText();
			cout << "Text prediction:" << textBasedPrediction() << endl;
			loadHistTestSample();
			compareHistograms();
			break;
		case 'r':
			reset = true;
			cout << "Reset " << endl;
			break;
		case 't':
			text = true;
			cout << "Text detection! " << endl;
			break;
		case 'c':
			if (cascade)
				cascade = false;
			else cascade = true;
			cout << "Cascade! " << endl;
			break;
		default:
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame, frameCount);
		imshow(window_name, frame);

		imshow(window_text, denoised);


		//cout << "\r" << frameCount<<" ";
		if (frameCount % 100) {

			R1 = avgVec(meanR1);
			R2 = avgVec(meanR2);
			R3 = avgVec(meanR3);
			R4 = avgVec(meanR4);
			R5 = avgVec(meanR5);

			if (R1 >= 0.02) {
				detectR1 = false;
			}

			if (R2 >= 0.02) {
				detectR2 = false;
			}

			if (R3 >= 0.02) {
				detectR3 = false;
			}

			if (R4 >= 0.02) {
				detectR4 = false;
			}

			if (R5 >= 0.02) {
				detectR5 = false;
			}

			//	std::cout << std::setprecision(4) << std::fixed << R1 << " " << R2 
			//<< " " << R3 << " " << R4 << " " << R5 << endl;
			meanR1.clear();
			meanR2.clear();
			meanR3.clear();
			meanR4.clear();
			meanR5.clear();
		}

		if (reset) {

			detectR1 = true;
			detectR2 = true;
			detectR3 = true;
			detectR4 = true;
			detectR5 = true;
			reset = false;
			t1 = false;
			t2 = false;
			t3 = false;
			t4 = false;
			t5 = false;
			h1 = false;
			h2 = false;
			h3 = false;
			h4 = false;
			h5 = false;
		}

		frameCount++;
	}
	return 0;
}



int main(int ac, char** av) {
	cv::CommandLineParser parser(ac, av, "{help h||}{@input||}");
	if (parser.has("help"))
	{
		help(av);
		system("pause");
		return 0;
	}
	std::string arg = parser.get<std::string>("@input");
	if (arg.empty()) {
		help(av);
		system("pause");
		return 1;
	}
	VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
	if (!capture.isOpened()) {
		cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
		help(av);
		return 1;
	}

	init();
	createSearchPattern();
	loadHistBase();
	return process(capture);
	system("pause");
}

void detectText(Mat frame, char * filename)
{
	std::string str;
	const char *result = "";
	sprintf(filename, "bin/data/input/input_image.png");
	imwrite(filename, frame);
	cout << "Saved " << filename << endl;
	str = "tesseract ";
	str += filename;
	str += " bin/data/output/output_text -psm 12";
	result = str.c_str();
	system(result);
}

void loadHistBase()
{
	Mat base1, base2, base3, base4, base5;
	Mat hsv_base1, hsv_base2, hsv_base3, hsv_base4, hsv_base5;

	base1 = imread("bin/data/hist_base/01.png", 1);
	base2 = imread("bin/data/hist_base/02.png", 1);
	base3 = imread("bin/data/hist_base/03.png", 1);
	base4 = imread("bin/data/hist_base/04.png", 1);
	base5 = imread("bin/data/hist_base/05.png", 1);

	/// Convert to HSV
	cvtColor(base1, hsv_base1, COLOR_BGR2HSV);
	cvtColor(base2, hsv_base2, COLOR_BGR2HSV);
	cvtColor(base3, hsv_base3, COLOR_BGR2HSV);
	cvtColor(base4, hsv_base4, COLOR_BGR2HSV);
	cvtColor(base5, hsv_base5, COLOR_BGR2HSV);

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_base1, 1, channels, Mat(), hist_base_1, 2, histSize, ranges, true, false);
	normalize(hist_base_1, hist_base_1, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_base2, 1, channels, Mat(), hist_base_2, 2, histSize, ranges, true, false);
	normalize(hist_base_2, hist_base_2, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_base3, 1, channels, Mat(), hist_base_3, 2, histSize, ranges, true, false);
	normalize(hist_base_3, hist_base_3, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_base4, 1, channels, Mat(), hist_base_4, 2, histSize, ranges, true, false);
	normalize(hist_base_4, hist_base_4, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_base5, 1, channels, Mat(), hist_base_5, 2, histSize, ranges, true, false);
	normalize(hist_base_5, hist_base_5, 0, 1, NORM_MINMAX, -1, Mat());

	cout << "Base histograms loaded!" << endl;
}
void loadHistTestSample()
{
	Mat test;
	Mat hsv_test;

	test = imread("bin/data/input/input_image.png", 1);

	/// Convert to HSV
	cvtColor(test, hsv_test, COLOR_BGR2HSV);

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_test, 1, channels, Mat(), hist_test_sample, 2, histSize, ranges, true, false);
	normalize(hist_test_sample, hist_test_sample, 0, 1, NORM_MINMAX, -1, Mat());


	cout << "Test histogram loaded!" << endl;
}
/// Histograms
/*
MatND hist_base_1;
MatND hist_base_2;
MatND hist_base_3;
MatND hist_base_4;
MatND hist_base_5;
MatND hist_test_sample;*/

int compareHistograms() {

	double base1_test = compareHist(hist_base_1, hist_test_sample, 0);
	double base2_test = compareHist(hist_base_2, hist_test_sample, 0);
	double base3_test = compareHist(hist_base_3, hist_test_sample, 0);
	double base4_test = compareHist(hist_base_4, hist_test_sample, 0);
	double base5_test = compareHist(hist_base_5, hist_test_sample, 0);

	double m = max(base1_test, base2_test);
	m = max(m, base3_test);
	m = max(m, base4_test);
	m = max(m, base5_test);

	if (m < 0.1) {
		cout << "Histogram Prediction: 0 " << m << endl;
		return 0;
	}

	else if (m == base1_test) {
		cout << "Histogram Prediction: 1 " << m << endl;
		h1 = true;
		return 1;
	}
	else if (m == base2_test) {
		cout << "Histogram Prediction: 2 " << m << endl;
		h2 = true;
		return 2;
	}
	else if (m == base3_test) {
		cout << "Histogram Prediction: 3 " << m << endl;
		h3 = true;
		return 3;
	}
	else if (m == base4_test) {
		cout << "Histogram Prediction: 4 " << m << endl;
		h4 = true;
		return 4;
	}
	else if (m == base5_test) {
		cout << "Histogram Prediction: 5 " << m << endl;
		h5 = true;
		return 5;
	}

}

vector <string> loadRawText()
{
	const char *filename = "bin/data/output/output_text.txt";
	ifstream myReadFile(filename);
	vector <string> rawData;
	string line;

	if (myReadFile.fail())
	{
		cout << "fail loading raw text!" << endl;
		myReadFile.close();
	}
	else {
		cout << "------------------------------------------------------------------" << endl;
		while (getline(myReadFile, line)) {

			rawData.push_back(line);
			cout << line << endl;
		}
		cout << "------------------------------------------------------------------" << endl;

	}
	myReadFile.close();
	cout << rawData.size() << " lines of text captured!" << endl;
	return rawData;
}

void createSearchPattern() {

	vector <string> item1;

	item1.push_back("bistouri");
	item1.push_back("BRAUN");
	item1.push_back("Blade");
	item1.push_back("bisturi");
	item1.push_back("Lama");
	item1.push_back("Lame");
	item1.push_back("Aesculap");

	itens.push_back(item1);

	vector <string> item2;

	item2.push_back("COVIDIEN");
	item2.push_back("AppOSe");
	item2.push_back("App");
	item2.push_back("AppOS");
	item2.push_back("Auto");
	item2.push_back("ULC");
	item2.push_back("Slim");
	item2.push_back("Body");
	item2.push_back("Skin");
	item2.push_back("Stapler");
	item2.push_back("8886803712");
	item2.push_back("88868");

	itens.push_back(item2);

	vector <string> item3;

	item3.push_back("Bastos");
	item3.push_back("Viegas");
	item3.push_back("Vieg");
	item3.push_back("bastosviegas");
	item3.push_back("4560-164");
	item3.push_back("Portugal");
	item3.push_back("4560");
	item3.push_back("298");

	itens.push_back(item3);

	vector <string> item4;

	item4.push_back("37");
	item4.push_back("9468");
	//item4.push_back("ETHICON");
	//item4.push_back("ETHIC");
	//item4.push_back("Coated");
	//item4.push_back("VICRYL");
	item4.push_back("Eur");
	item4.push_back("braided");
	item4.push_back("absorbable");
	item4.push_back("suture");
	item4.push_back("Poly");
	item4.push_back("2022");
	item4.push_back("suture");
	//item4.push_back("90cm");

	itens.push_back(item4);

	vector <string> item5;

	item5.push_back("34");
	item5.push_back("9464");
	//item5.push_back("ETHICON");
	//item5.push_back("ETHIC");
	//item5.push_back("Coated");
	//item5.push_back("VICRYL");
	//item5.push_back("Eur");
	//item5.push_back("braided");
	//item5.push_back("absorbable");
	//item5.push_back("suture");
	//item5.push_back("Poly");
	//item5.push_back("2022");
	//item5.push_back("suture");
	//item5.push_back("90cm");

	itens.push_back(item5);

	cout << "item definitions loaded" << endl;

}

int findWordInText(int itemPosition) {

	string savedStrings = "";
	int returnInt = 0;
	vector <string> tempLine = itens[itemPosition];

	for (int s = 0; s < itens[itemPosition].size(); s++) {

		string searchString = itens[itemPosition][s];

		for (int i = 0; i < rawTextDataVec.size(); i++) {

			string lineString = rawTextDataVec.at(i);

			if (findStringIC(lineString, searchString)) {
				returnInt++;
				savedStrings += searchString + " ";
				break;
			}
		}
	}

	cout << returnInt << " " << savedStrings << endl;
	return returnInt;
}

/// Try to find in the Haystack the Needle - ignore case
bool findStringIC(const std::string & strHaystack, const std::string & strNeedle)
{
	auto it = std::search(
		strHaystack.begin(), strHaystack.end(),
		strNeedle.begin(), strNeedle.end(),
		[](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
	);
	return (it != strHaystack.end());
}

int textBasedPrediction() {
	int result = 0;
	int i = 0;

	while (i < 5) {

		cout << "searching item: " << i + 1 << endl;
		result = findWordInText(i);

		if (result > 0) {
			result = i + 1;

			if (result == 1) t1 = true;
			if (result == 2) t2 = true;
			if (result == 3) t3 = true;
			if (result == 4) t4 = true;
			if (result == 5) t5 = true;
			break;
		}
		i++;
	}
	return result;
}

void detectAndDisplay(Mat frame, int frameCount)
{
	std::vector<Rect> rectR1;       // rotule 1
	std::vector<Rect> rectR2;		// rotule 2
	std::vector<Rect> rectR3;       // rotule 3
	std::vector<Rect> rectR4;		// rotule 4
	std::vector<Rect> rectR5;       // rotule 5

	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	Point2f a(3, 3), b(140, 90);
	rectangle(frame, a, b, (0, 0, 0), CV_FILLED, 8, 0);
	Point2f c(frame.cols / 4, frame.rows / 4), d((frame.cols / 4) * 3, (frame.rows / 4) * 3);
	rectangle(frame, c, d, Scalar(0, 0, 0), 0.5, 4, 0);

	if (t1) {
		putText(frame, "TR1 ", Point(50, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1.5, 8, false);
	}
	if (t2) {
		putText(frame, "TR2 ", Point(50, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1.5, 8, false);
	}
	if (t3) {
		putText(frame, "TR3 ", Point(50, 45), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
	}

	if (t4) {
		putText(frame, "TR4 ", Point(50, 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5, 8, false);
	}
	if (t5) {
		putText(frame, "TR5 ", Point(50, 75), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
	}

	if (h1) {
		putText(frame, "HR1 ", Point(90, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1.5, 8, false);
	}
	if (h2) {
		putText(frame, "HR2 ", Point(90, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1.5, 8, false);
	}
	if (h3) {
		putText(frame, "HR3 ", Point(90, 45), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
	}

	if (h4) {
		putText(frame, "HR4 ", Point(90, 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5, 8, false);
	}
	if (h5) {
		putText(frame, "HR5 ", Point(90, 75), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
	}

	if (cascade) {
		if (detectR1) {
			r1_cascade.detectMultiScale(frame_gray, rectR1, 1.1, 1, 0, Size(80, 80), Size(120, 120));
			for (size_t i = 0; i < rectR1.size(); i++)
			{
				Point center(rectR1[i].x + rectR1[i].width / 2, rectR1[i].y + rectR1[i].height / 2);
				ellipse(frame, center, Size(rectR1[i].width / 2, rectR1[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
				putText(frame, "CR1 ", Point(10, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1.5, 8, false);
				meanR1.push_back(1.0);
			}
		}
		else {
			putText(frame, "CR1 ", Point(10, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1.5, 8, false);
		}

		if (detectR2) {
			r2_cascade.detectMultiScale(frame_gray, rectR2, 1.1, 1, 0, Size(80, 80), Size(300, 300));
			for (size_t i = 0; i < rectR2.size(); i++)
			{
				Point center2(rectR2[i].x + rectR2[i].width / 2, rectR2[i].y + rectR2[i].height / 2);
				ellipse(frame, center2, Size(rectR2[i].width / 2, rectR2[i].height / 2), 0, 0, 360, Scalar(255, 255, 0), 4, 8, 0);
				putText(frame, "CR2 ", Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1.5, 8, false);
				meanR2.push_back(1.0);
			}
		}
		else {
			putText(frame, "CR2 ", Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1.5, 8, false);
		}

		if (detectR3) {
			r3_cascade.detectMultiScale(frame_gray, rectR3, 1.1, 1, 0, Size(100, 100), Size(300, 300));
			for (size_t i = 0; i < rectR3.size(); i++)
			{
				Point center3(rectR3[i].x + rectR3[i].width / 2, rectR3[i].y + rectR3[i].height / 2);
				ellipse(frame, center3, Size(rectR3[i].width / 2, rectR3[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
				putText(frame, "CR3 ", Point(10, 45), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
				meanR3.push_back(1.0);
			}
		}
		else {
			putText(frame, "CR3 ", Point(10, 45), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5, 8, false);
		}

		if (detectR4) {
			r4_cascade.detectMultiScale(frame_gray, rectR4, 1.1, 1, 0, Size(100, 100), Size(300, 300));
			for (size_t i = 0; i < rectR4.size(); i++)
			{
				Point center4(rectR4[i].x + rectR4[i].width / 2, rectR4[i].y + rectR4[i].height / 2);
				ellipse(frame, center4, Size(rectR4[i].width / 2, rectR4[i].height / 2), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);
				putText(frame, "CR4 ", Point(10, 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5, 8, false);
				meanR4.push_back(1.0);
			}
		}
		else {
			putText(frame, "CR4 ", Point(10, 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5, 8, false);
		}

		if (detectR5) {
			r5_cascade.detectMultiScale(frame_gray, rectR5, 1.1, 1, 0, Size(100, 100), Size(300, 300));
			for (size_t i = 0; i < rectR5.size(); i++)
			{
				Point center5(rectR5[i].x + rectR5[i].width / 2, rectR5[i].y + rectR5[i].height / 2);
				ellipse(frame, center5, Size(rectR5[i].width / 2, rectR5[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
				putText(frame, "CR5 ", Point(10, 75), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1.5, 8, false);
				meanR5.push_back(1.0);
			}
		}
		else {
			putText(frame, "CR5 ", Point(10, 75), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1.5, 8, false);
		}
		if (!detectR1 && !detectR2 && !detectR3 && !detectR4 && !detectR5) {

			putText(frame, "Complete!", Point(10, 90), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.5, 8, false);
		}

	}

}

double avgVec(const vector<double> & v)
{
	double sum = 0;

	for (double x : v) sum += x;

	return v.empty() ? 0 : sum / 100;
}


