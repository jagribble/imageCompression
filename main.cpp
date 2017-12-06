#include <opencv2/opencv.hpp>
#include "Node.h"
#include "bitstring.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


// dft
Mat makeDFT(Mat I,const char *name);
void swapQuadrants(Mat& img);

void dImage(Mat& image, Mat& output,int flag);

void compress(Mat& img);
void de_Compress();
void threeChannels(Mat& img, Mat& red, Mat& green, Mat& blue, Mat* rgbArray);
void huffman(Mat img);
void addPrefix(PNode *root, String prefix, map<int,string> &huffmanTable);
void getOutputString(map<int,string> &huffmanTable,Mat &img, string &outputString);
void sortHuffman(vector<PNode>& array);


Mat huffmanDecode();
void getValue(PNode *root, string &binary);

vector<PNode> priorityQueue;
vector<int> decodedValues;
int width,height;

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
    Mat image = imread("../2small.ppm",CV_LOAD_IMAGE_COLOR);
    //Mat image = imread("../fish.jpg",CV_LOAD_IMAGE_COLOR);
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
    huffman(image);
   // huffmanDecode();
    de_Compress();
    imwrite("../decomp.jpg",image);
    dctImages[0].convertTo(dctImages[0],CV_8U);

//    Mat smallImage = Mat(image,Rect(0,0,110,70));
//
//    imshow("small block", smallImage);
//    waitKey();
    //makeDFT(rgbArray[0],"dft");


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

void de_Compress(){
    Mat img = huffmanDecode();
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
    width = img.size().width;
    height = img.size().height;
    // go through all the pixels in the image and get the value
    // if the value is already stored then add 1 to it's frequency
    vector<Mat> channels;
    split(img,channels);
    for(int x=0;x<img.rows;x++){
        for(int y=0;y<img.cols;y++){

            for(int channel=0;channel<channels.size();channel++){

                int value = int(channels.at(channel).at<uchar>(x,y));


                if(find(numbers.begin(), numbers.end(), value) != numbers.end()){
                    // add frequency
                    int pos = position(numbers,value);

                    freq.at(pos) = freq.at(pos) +1;

                } else{
                    numbers.push_back(int(channels.at(channel).at<uchar>(x,y)));
                    freq.push_back(1);
                }
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

    for(int x=0;x<freq.size();x++){
        int pixels = img.rows*img.cols;
        PNode node = PNode(numbers.at(x),freq.at(x),pixels);
        priorityQueue.push_back(node);
    }
    cout<<"first in the array is vale=->"<<priorityQueue.at(0).value <<endl;
    // sort the priority queue to get it in lowest priority at value 0
    sortHuffman(priorityQueue);
    float totalValue = 0;
    //make huffman encoding tree
    while(totalValue<1){
        // get the first two nodes and put them on the left and right of a new parent node
        PNode left = priorityQueue.at(0);
        PNode right = priorityQueue.at(1);
        cout << "Left: "<<left.huffmanProbability<<" right: "<<right.huffmanProbability<<endl;
        // add the frequency left and right probabilities
        float newPriority = left.huffmanProbability + right.huffmanProbability;
        PNode *parentNode = new PNode();
        parentNode->huffmanProbability = newPriority;
        parentNode->left = new PNode(left.value,left.frequency,left.left,left.right,left.prefix,left.huffmanProbability);
        parentNode->right = new PNode(right.value,right.frequency,right.left,right.right,right.prefix,right.huffmanProbability);;
        // add the parent node to the priority queue and then sort the list again
        priorityQueue.erase(priorityQueue.begin());
        priorityQueue.at(0) = *parentNode;
        sortHuffman(priorityQueue);
        // when root node is made it will have priority 1
        totalValue =newPriority;
    //    cout << parentNode->left->value << endl;
        cout<< "priority -->"<< newPriority <<endl;

    }

    map<int,string> huffmanTable;

    addPrefix(&priorityQueue.at(0),"",huffmanTable);
    string outputString;
    getOutputString(huffmanTable,img,outputString);
    string charString;
   // getChars(outputString,charString);
    //ostream stream(std::move(std::ostringstream()));
    stringstream sstream(outputString);
    while (sstream.good()){
        std::bitset<8> bits;
        sstream >> bits;
        char c = char(bits.to_ulong());
        charString += c;
    }
    ofstream outData("output.txt");

    outData << charString;

}

void addPrefix(PNode *root, String prefix, map<int,string> &huffmanTable){
    //PNode node = root;
    if(root->value != -1){
        // leaf node
        cout << root->value <<" : "<<prefix<<endl;
        PNode node = *root;
        root->prefix = prefix;
        huffmanTable[root->value] = prefix;
        //&root->value->setPrefix(prefix);
        return;
    }
    else{
        addPrefix(root->left, prefix + "0",huffmanTable);
        addPrefix(root->right, prefix + "1",huffmanTable);
    }
}

void getOutputString(map<int,string> &huffmanTable,Mat &img, string &outputString ){
    vector<Mat> channels;
    split(img,channels);
    for(int x=0;x<img.rows;x++){
        // loop through each row for each channel
        for (int channel = 0; channel < channels.size(); ++channel) {
            for(int y=0;y<img.cols;y++){
                outputString += huffmanTable[int(channels.at(channel).at<uchar>(x,y))];
            }
            // for each channel append the channel type to the end of the string to tell which channel
//            if (channel == 0){
//                outputString += "/y";
//            } else if (channel == 1){
//                outputString += "/cr";
//            } else{
//                outputString += "/cb";
//            }
        }
    }

}


Mat huffmanDecode(){
    ifstream file;
    file.open("output.txt");
    char c;
    string binaryString;
    while (file.get(c))  {
        // loop getting single characters
        bitset<8> byte = c;
        binaryString += byte.to_string();
      //  std::cout << byte;
    }
  //  cout << binaryString;
    while(!binaryString.empty()){
        getValue(&priorityQueue.at(0),binaryString);
    }
    cout<<"Finished"<<endl;
    int arrayPos = 0;
    Mat img = Mat(height,width,CV_8UC3);
    cvtColor(img,img,COLOR_BGR2YCrCb);
    for(int x=0;x<img.rows;x++){
        vector<Mat> channels;
        split(img, channels);
        for(int channel = 0; channel < channels.size(); channel++){
            for(int y=0;y<img.cols;y++){
                channels[channel].at<uchar>(x,y) = static_cast<uchar>(decodedValues[arrayPos]);
                arrayPos++;
            }
        }
        merge(channels,img);

    }
    imshow("image after decoded",img);
    waitKey();

    return img;


}


void getValue(PNode *root, string &binary){
    // if there is value then reached leaf node and output to vector
    if(root->value != -1){
        // leaf node
        //cout << "Leaf node value : " << root->value << endl;
        decodedValues.push_back(root->value);
//        cout << root->value <<" : "<<prefix<<endl;
//        PNode node = *root;
//        root->prefix = prefix;
       // huffmanTable[root->value] = prefix;
        //&root->value->setPrefix(prefix);
        return;
    }
    else{
        if(!binary.empty() && binary.at(0) == '0'){
            binary.erase(0,1);
            getValue(root->left, binary);
        } else if(!binary.empty() && binary.at(0) == '1'){
            binary.erase(0,1);
            getValue(root->right, binary);
        }
        return;
    }
}
