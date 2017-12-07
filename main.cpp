#include <opencv2/opencv.hpp>
#include "Node.h"
#include "bitstring.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


void compress(Mat& img);
void de_Compress(Mat &img);

int position(std::vector<int> numbers, int value);
void sortFreq(std::vector<int>& numbers, std::vector<int>& freq);
void huffman(Mat img);
void addPrefix(PNode *root, String prefix, map<int,string> &huffmanTable);
void getOutputString(map<int,string> &huffmanTable,Mat &img, string &outputString);
void sortHuffman(vector<PNode>& array);


Mat huffmanDecode();
void getValue(PNode *root, string &binary);

void compressionRatio(string input);
void meanSquareError(Mat &original, Mat &decompressed);

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
    string inputString = "../2small.ppm";
    std::cout << "Hello, World!" << std::endl;
    Mat image = imread(inputString,CV_LOAD_IMAGE_COLOR);
    //Mat image = imread("../fish.jpg",CV_LOAD_IMAGE_COLOR);

    compress(image);
    Mat imageChannels[3];
    split(image,imageChannels);
    imwrite("../output.jpg",image);
    huffman(image);
    Mat decompressedImage;
    de_Compress(decompressedImage);
    compressionRatio(inputString);
    meanSquareError(image,decompressedImage);

    return 0;
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

    namedWindow("original",CV_WINDOW_AUTOSIZE);
    imshow("original",img);
    waitKey();
    // convert Image from BGR to YCrCb
    cvtColor(img,img,COLOR_BGR2YCrCb);
    cout<<"cols->"<< img.cols<<endl;
    cout<<"x->"<< img.rows<<endl;
    // loop through the image and split into blocks.
    for(int x=0;x<img.rows;x+=8){
        for(int y=0;y<img.cols;y+=8){

            if((x+8)<img.rows-1 && (y+8) <img.cols-1){
                Mat block = img(Rect(y,x,8,8));
                vector<Mat> channels;
                split(block,channels);
                vector<Mat> outputChannels(channels.size());
                //for each channel of the block
                for(int p=0;p<block.channels();p++){
                    Mat channelBlock = channels[p];
                    channelBlock.convertTo(channelBlock,CV_64FC1);
                    //subtract 128
                    subtract(channelBlock, 128.0, channelBlock);
                    // DCT on the block for the channel
                    Mat blockDCT;
                    dct(channelBlock,channelBlock);
                    // divide by the relevant quantization table
                    if(p==0){
                        //luminance
                        divide(channelBlock,lum,channelBlock);
                    }else{
                        divide(channelBlock,chrom,channelBlock);
                    }
                    // add 128 to the block
                    add(channelBlock, 128.0, channelBlock);
                    channelBlock.convertTo(channelBlock,CV_8UC1);
                    outputChannels[p] = channelBlock;

                }
                // merge all channels for the block
                merge(outputChannels,block);

            }

        }

    }
    //show the image
    namedWindow("DCT + quantization ",CV_WINDOW_AUTOSIZE);
    imshow("DCT + quantization ",img);
    waitKey();

}

void de_Compress(Mat &img){
    // Get the data from the text file and traverse the huffman encoding tree to get the relevant data
    img = huffmanDecode();
    Mat lum = Mat(8,8,CV_64FC1,&dataLum);
    Mat chrom = Mat(8,8,CV_64FC1,&dataChrom);

  //  imshow("into decompression",img);
    // loop though each pixel of the image and divide it into blocks
    for(int x=0;x<img.rows;x+=8){
        for(int y=0;y<img.cols;y+=8){

            if((x+8)<img.rows-1 && (y+8) <img.cols-1){
                Mat block = img(Rect(y,x,8,8));
                vector<Mat> channels;
                split(block,channels);
                vector<Mat> outputChannels(channels.size());
                // seperate the image by channels
                for(int p=0;p<block.channels();p++){
                    Mat channelBlock = channels[p];
                    channelBlock.convertTo(channelBlock,CV_64FC1);
                    // subtract 128 from the block for this channel
                    subtract(channelBlock, 128.0, channelBlock);
                    // multiply the block by the relevant quantization table
                    if(p==0){
                        //luminance
                        multiply(channelBlock,lum,channelBlock);
                    }else{
                        multiply(channelBlock,chrom,channelBlock);
                    }

                    // peform inverse DCT on the block
                    Mat blockDCT;
                    idct(channelBlock,blockDCT);
                    // add 128 to the block
                    add(blockDCT, 128.0, blockDCT);
                    blockDCT.convertTo(blockDCT,CV_8UC1);

                    outputChannels[p] = blockDCT;

                }
                // merge all the channels to the block
                merge(outputChannels,block);

            }

        }

    }
    // Convert YCrCb to BGR
    cvtColor(img,img,COLOR_YCrCb2BGR);
    // show decompressed image
    namedWindow("decompressed",CV_WINDOW_AUTOSIZE);
    imshow("decompressed",img);
    waitKey();
    imwrite("../decomp.ppm",img);
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

/* Sort the huffman priority vector through insertion sort*/
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
    // vector of the numbers (intensity values) and frequencies of those (direct mapping)
    std::vector<int> numbers =  std::vector<int>();
    std::vector<int> freq =  std::vector<int>();
    // set the width and height of the image to the global varibles
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
        int pixels = img.rows*img.cols*3;
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
    // add the prefixes to each value and put it in the huffman table
    addPrefix(&priorityQueue.at(0),"",huffmanTable);
    string outputString;
    // get the bits from the huffman table for each pixels
    getOutputString(huffmanTable,img,outputString);
    string charString;
    // get the bits and for each 8 bits (1 byte) get the character and store it to file
    stringstream sstream(outputString);
    while (sstream.good()){
        std::bitset<8> bits;
        sstream >> bits;
        char c = char(bits.to_ulong());
        charString += c;
    }
    // write the data to file
    ofstream outData("output.txt");

    outData << charString;
    outData.close();

}

void addPrefix(PNode *root, String prefix, map<int,string> &huffmanTable){
    // if there is a value then it must be a leaf node so add the prefix to the huffman table
    if(root->value != -1){
        // leaf node
        cout << root->value <<" : "<<prefix<<endl;
        PNode node = *root;
        root->prefix = prefix;
        huffmanTable[root->value] = prefix;
        return;
    }
    else{
        // if the value is -1 then go left and right.
        addPrefix(root->left, prefix + "0",huffmanTable);
        addPrefix(root->right, prefix + "1",huffmanTable);
    }
}

void getOutputString(map<int,string> &huffmanTable,Mat &img, string &outputString ){
    vector<Mat> channels;
    split(img,channels);
    // get the output string
    for(int x=0;x<img.rows;x++){
        // loop through each row for each channel get the bit values from the huffman table for that inensity value
        for (int channel = 0; channel < channels.size(); ++channel) {
            for(int y=0;y<img.cols;y++){
                outputString += huffmanTable[int(channels.at(channel).at<uchar>(x,y))];
            }

        }
    }

}


Mat huffmanDecode(){
    // decode the file
    ifstream file;
    // open the file
    file.open("output.txt");
    char c;
    string binaryString;
    while (file.get(c))  {
        // loop getting single characters
        bitset<8> byte = c;
        binaryString += byte.to_string();

    }
    // if the string is not empty get the value from the binary string
    while(!binaryString.empty()){
        getValue(&priorityQueue.at(0),binaryString);
    }
    cout<<"Finished"<<endl;

    int arrayPos = 0;
    Mat img = Mat(height,width,CV_8UC3);
    cvtColor(img,img,COLOR_BGR2YCrCb);
    // go through the image and get the value for each channel on that row.
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
    namedWindow("image after decoded",CV_WINDOW_AUTOSIZE);
    imshow("image after decoded",img);
    waitKey();

    return img;


}


void getValue(PNode *root, string &binary){
    // if there is value then reached leaf node and output to vector
    if(root->value != -1){
        // leaf node
        decodedValues.push_back(root->value);

        return;
    }
    else{
        // if it's a 0 then go left
        if(!binary.empty() && binary.at(0) == '0'){
            binary.erase(0,1);
            getValue(root->left, binary);
            // if 1 then go right
        } else if(!binary.empty() && binary.at(0) == '1'){
            binary.erase(0,1);
            getValue(root->right, binary);
        }
        return;
    }
}


void compressionRatio(string input){
    // get input and output file and seek to the end of file
    ifstream inputFile(input,ifstream::in | ifstream::binary);
    inputFile.seekg(0, ios::end);
    ifstream outputFile("output.txt",ifstream::in | ifstream::binary);
    outputFile.seekg(0, ios::end);
    // work out the compression ratio and show to screen
    float compressionRatio = inputFile.tellg()/outputFile.tellg();
    cout << "original file size -> "<<inputFile.tellg()<<endl;
    cout << "Compressed file size -> "<<outputFile.tellg()<<endl;
    cout << "Compression Ratio = " << compressionRatio<< endl;
    inputFile.close();
    outputFile.close();
}

// check the mean square error of all channels. MSE of 0 means it is the same as the original image (lossless)
void meanSquareError(Mat &original, Mat &decompressed){
    int sum = 0;
    int no = 0;
    for (int x=0;x<original.rows;x++){
        // split the original and compressed images into different channels
        vector<Mat> originalChannels;
        split(original,originalChannels);
        vector<Mat> decompressedChannels;
        split(decompressed,decompressedChannels);
        for(int channel=0;channel<originalChannels.size();channel++){
            // for each pixel on the row add to the sum the different^2 of each pixel
            for(int y=0;y<original.cols;y++){
                int difference = originalChannels.at(channel).at<uchar>(x,y)-decompressedChannels.at(channel).at<uchar>(x,y);
                sum+= pow(difference,2);
                no++;
            }
        }
    }
    float mse = sum/(original.rows * original.cols*3);
    cout << "MSE = " << mse <<endl;

}