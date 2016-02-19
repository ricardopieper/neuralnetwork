#include "FlensOctave.hpp"

using namespace std;

/*
 *  string::size_type loc = line.find(type_header);
            if (loc != string::npos){
                string type = line.substr(loc+type_header.length(), length-loc);

 *
 * */

string getConfig(ifstream& ifs, std::string conf){

    string line;
    getline(ifs,line);

    std::string last = line.substr(line.length()-1, 1);

    if (last == "\r"){
        line = line.substr(0, line.length() - 1);
    }

    string::size_type startingPos = line.find(conf);

    if (startingPos != string::npos){

        string::size_type infoStartPos = startingPos + conf.length();
        string::size_type endPos = line.length();

        string::size_type infoLength = endPos - infoStartPos;

        string info = line.substr(infoStartPos, infoLength);
        return info;
    }else{
        return "";
    }
}


Mtx* FlensOctave::LoadMatrix(const char* fullPath){

  ifstream ifs(fullPath);
  string line;
  if (ifs.is_open()){

     //format expected:
    //header
    // #name: something
    // #type: matrix
    // #rows: number
    // #columns: number

    getline(ifs, line);
    getline(ifs, line);

    string typeConf = getConfig(ifs, "# type: ");

    if (typeConf != "" && typeConf == "matrix"){

        int rows = std::atoi(getConfig(ifs, "# rows: ").c_str());
        int cols = std::atoi(getConfig(ifs, "# columns: ").c_str());

        Mtx* m = new Mtx(rows, cols);
        int i =1, j;

        while (getline(ifs, line)) {

            if (line == "") break;

            j = 1;

            std::string currentNumber = "";
            std::string last = line.substr(line.length()-1, 1);

            if (last == "\r"){
                line = line.substr(0, line.length() - 1);
            }
            for (char &c: line){

                if (c == ' '){

                    if (currentNumber != ""){
                        double parsed = std::stod(currentNumber);
                        (*m)(i,j) = parsed;
                        j++;
                        currentNumber = "";
                    }
                }
                else
                {
                    currentNumber += c;
                }

            }

            if (currentNumber != "" && currentNumber != "\r"){
               double parsed = std::stod(currentNumber);
                (*m)(i,j) = parsed;

            }
            i++;
        }

        return m;

    }
    else return NULL;
  }

  return NULL;
}
