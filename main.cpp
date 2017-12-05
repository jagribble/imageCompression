#include <opencv2/opencv.hpp>
#include "Node.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


// dft
Mat makeDFT(Mat I,const char *name);
void swapQuadrants(Mat& img);

void dImage(Mat& image, Mat& output,int flag);

void compress(Mat& img);
void de_Compress(Mat& img);
void threeChannels(Mat& img, Mat& red, Mat& green, Mat& blue, Mat* rgbArray);
void huffman(Mat img);
void addPrefix(PNode *root, String prefix);
void sortHuffman(vector<PNode>& array);

double dataLum[8][8] = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
};
//    int data[8][8] = {
//            {1, 1, 1, 1, 0, 0, 0, 0},
//            {1, 1, 1, 0, 0, 0, 0, 0},
//            {1, 1, 0, 0, 0, 0, 0, 0},
//            {1, 0, 0, 0, 0, 0, 0, 0},
//            {0, 0, 0, 0, 0, 0, 0, 0},
//            {0, 0, 0, 0, 0, 0, 0, 0},
//            {0, 0, 0, 0, 0, 0, 0, 0},
//            {0, 0, 0, 0, 0, 0, 0, 0}
//    };
double dataChrom[8][8] = {
        {17, 18, 24, 27, 99, 99, 99, 99},
        {18, 21, 26, 66, 99, 99, 99, 99},
        {24, 26, 56, 99, 99, 99, 99, 99},
        {47, 66, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat image = imread("../2.ppm",CV_LOAD_IMAGE_COLOR);
   // Mat image = imread("../fish.jpg",CV_LOAD_IMAGE_COLOR);
    imshow("fire",image);
    waitKey();
    Mat rgbArray[3];

    Mat result_red(image.rows, image.cols, CV_8UC3);
    Mat result_green(image.rows, image.cols, CV_8UC3);
    Mat result_blue(image.rows, image.cols, CV_8UC3);
    split(image,rgbArray);
    threeChannels(image,result_red,result_green,result_blue,rgbArray);
    /*imshow("fire red",result_red);
    waitKey();
    imshow("fire green",result_green);
    waitKey();
    imshow("fire blue",result_blue);
    waitKey();*/

    Mat dctImages[3];
    for(int x=0;x<3;x++){
        Mat dctImage;
        Mat inImage = rgbArray[x];

        dImage(inImage,dctImage,0);
//        imshow("dct image", dctImage);
//        waitKey();
        dctImages[x] = dctImage;
        Mat inverse;
        dImage(dctImage,inverse,1);
//        imshow("dct inverse",inverse);
//        waitKey();
    }
    compress(image);
    Mat imageChannels[3];
    split(image,imageChannels);
    imwrite("../output.jpg",image);
    de_Compress(image);
    imwrite("../decomp.jpg",image);
    dctImages[0].convertTo(dctImages[0],CV_8U);

//    Mat smallImage = Mat(image,Rect(0,0,110,70));
//
//    imshow("small block", smallImage);
//    waitKey();
    //makeDFT(rgbArray[0],"dft");

    huffman(imageChannels[0]);
    return 0;
}

void threeChannels(Mat& img, Mat& red, Mat& green, Mat& blue, Mat* rgbArray){
    Mat empty_image = Mat::zeros(img.size(), CV_8UC1);
    Mat inBlue[] =  { rgbArray[0], empty_image, empty_image};
    Mat inGreen[] =  { empty_image, rgbArray[1], empty_image};
    Mat inRed[] = { empty_image, empty_image, rgbArray[2]};
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(inBlue,3,&blue,1,from_to,3);
    mixChannels(inGreen,3,&green,1,from_to,3);
    mixChannels( inRed, 3, &red, 1, from_to, 3 );

}

int position(std::vector<int> numbers, int value){
    int x=0;
    for(;x<numbers.size();x++){
        if(numbers.at(x)==value){
            break;
        }
    }
    return x;
}

void sortFreq(std::vector<int>& numbers, std::vector<int>& freq){
    int j, tempFreq, tempNum;

    for (int i = 0; i < numbers.size(); i++){
        j = i;

        while (j > 0 && freq[j] < freq[j-1]){
            tempFreq = freq[j];
            tempNum = numbers[j];
            freq[j] = freq[j-1];
            numbers[j] = numbers[j-1];
            freq[j-1] = tempFreq;
            numbers[j-1] = tempNum;
            j--;
        }
    }
}

void compress(Mat& img){
    //img.convertTo(img,CV_32F);
//
    Mat lum = Mat(8,8,CV_64FC1,&dataLum);
    Mat chrom = Mat(8,8,CV_64FC1,&dataChrom);

    imshow("orig",img);
    waitKey();
    cvtColor(img,img,COLOR_BGR2YCrCb);
    cout<<"cols->"<< img.cols<<endl;
    cout<<"x->"<< img.rows<<endl;
    for(int x=0;x<img.rows;x+=8){
        for(int y=0;y<img.cols;y+=8){

            if((x+8)<img.rows-1 && (y+8) <img.cols-1){
                Mat block = img(Rect(y,x,8,8));
                vector<Mat> channels;
                split(block,channels);
                vector<Mat> outputChannels(channels.size());
                for(int p=0;p<block.channels();p++){
                    Mat channelBlock = channels[p];
                    channelBlock.convertTo(channelBlock,CV_64FC1);
                    subtract(channelBlock, 128.0, channelBlock);

                    Mat blockDCT;
                    dct(channelBlock,channelBlock);

                    if(p==0){
                        //luminance
                        divide(channelBlock,lum,channelBlock);
                    }else{
                        divide(channelBlock,chrom,channelBlock);
                    }
                    add(channelBlock, 128.0, channelBlock);
                    channelBlock.convertTo(channelBlock,CV_8UC1);
                    outputChannels[p] = channelBlock;

                }
                merge(outputChannels,block);

            }

        }
      //  cout << x<<endl;
    }


   // merge(smallimages,img);
    imshow("q",img);
    waitKey();

}

void de_Compress(Mat& img){

    Mat lum = Mat(8,8,CV_64FC1,&dataLum);
    Mat chrom = Mat(8,8,CV_64FC1,&dataChrom);

    imshow("into decompression",img);

    for(int x=0;x<img.rows;x+=8){
        for(int y=0;y<img.cols;y+=8){

            if((x+8)<img.rows-1 && (y+8) <img.cols-1){
                Mat block = img(Rect(y,x,8,8));
                vector<Mat> channels;
                split(block,channels);
                vector<Mat> outputChannels(channels.size());
                for(int p=0;p<block.channels();p++){
                    Mat channelBlock = channels[p];
                    channelBlock.convertTo(channelBlock,CV_64FC1);
                    subtract(channelBlock, 128.0, channelBlock);
                    if(p==0){
                        //luminance
                        multiply(channelBlock,lum,channelBlock);
                    }else{
                        multiply(channelBlock,chrom,channelBlock);
                    }


                    Mat blockDCT;
                    idct(channelBlock,blockDCT);
                    add(blockDCT, 128.0, blockDCT);
                    blockDCT.convertTo(blockDCT,CV_8UC1);

                    outputChannels[p] = blockDCT;

                }
                merge(outputChannels,block);

            }

        }
        //cout << x<<endl;
    }
    cvtColor(img,img,COLOR_YCrCb2BGR);

    imshow("inverse",img);
    waitKey();
}


void dImage(Mat& image, Mat& output,int flag){
    //if flag set then inverse
    if(flag == 1){
        dct(image,output,DCT_INVERSE);
        output.convertTo(output,CV_8UC1);
    } else{
        image.convertTo(image,CV_32F);
        dct(image,output);
    }

}

void sortHuffman(vector<PNode>& array){
    int j;
    PNode temp;

    for (int i = 0; i < array.size(); i++){
        j = i;

        while (j > 0 && array[j].huffmanProbability < array[j-1].huffmanProbability){
            temp = array[j];

            array[j] = array[j-1];

            array[j-1] = temp;

            j--;
        }
    }
}


void huffman(Mat img){
    std::vector<int> numbers =  std::vector<int>();
    std::vector<int> freq =  std::vector<int>();
    // go through all the pixels in the image and get the value
    // if the value is already stored then add 1 to it's frequency
    for(int x=0;x<img.rows;x++){
        for(int y=0;y<img.cols;y++){
            int value = int(img.at<uchar>(x,y));
           // std::cout << value << std::endl;

            if(find(numbers.begin(), numbers.end(), value) != numbers.end()){
                // add frequency
                int pos = position(numbers,value);
           //     std::cout << value << " at " << pos << " vector size >>>"<<numbers.size()<<std::endl;
                freq.at(pos) = freq.at(pos) +1;

            } else{
                numbers.push_back(int(img.at<uchar>(x,y)));
                freq.push_back(1);
            }

        }
    }
    //sort the frquencies
    sortFreq(numbers,freq);
    std::cout<<"Vector size: "<<numbers.size() <<std::endl;
    std::cout << "________________"<<std::endl;
    std::cout << "| number | freq.|"<<std::endl;
    for(int x= int(numbers.size() - 1); x >= 0; x--){
        std::cout << "| "<<numbers.at(x)<<" | "<<freq.at(x)<<" |"<<std::endl;
    }
    // create nodes for each value
    vector<PNode> priorityQueue;
    for(int x=0;x<freq.size();x++){
        int pixels = img.rows*img.cols;
        PNode node = PNode(numbers.at(x),freq.at(x),pixels);
        priorityQueue.push_back(node);
    }
    cout<<"first in the array is vale=->"<<priorityQueue.at(0).value <<endl;
    sortHuffman(priorityQueue);
    float totalValue = 0;
    //make huffman encoding tree
    while(totalValue<1){
        PNode left = priorityQueue.at(0);
        PNode right = priorityQueue.at(1);
        cout << "Left: "<<left.huffmanProbability<<" right: "<<right.huffmanProbability<<endl;
        float newPriority = left.huffmanProbability + right.huffmanProbability;
        PNode *parentNode = new PNode();
        parentNode->huffmanProbability = newPriority;
        parentNode->left = new PNode(left.value,left.frequency,left.left,left.right,left.prefix,left.huffmanProbability);
        parentNode->right = new PNode(right.value,right.frequency,right.left,right.right,right.prefix,right.huffmanProbability);;
        priorityQueue.erase(priorityQueue.begin());
        priorityQueue.at(0) = *parentNode;
        sortHuffman(priorityQueue);
        // when root node is made it will have priority 1
        totalValue =newPriority;
    //    cout << parentNode->left->value << endl;
        cout<< "priority -->"<< newPriority <<endl;

    }

    addPrefix(&priorityQueue.at(0),"");

}

void addPrefix(PNode *root, String prefix){
    //PNode node = root;
    if(root->value != -1){
        // leaf node
        cout << root->value <<" : "<<prefix<<endl;
        return;
    }
    else{
        addPrefix(root->left, prefix + "0");
        addPrefix(root->right, prefix + "1");
    }
}


