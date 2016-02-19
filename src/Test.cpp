// Copyright 2016 Ricardo Pieper


#include <iostream>

#include "FlensOctave.hpp"
#include "DigitClassifier.hpp"
using namespace std;


int main() {
    FlensOctave v;

    Mtx * X = v.LoadMatrix("data/dataset");
    Mtx * y = v.LoadMatrix("data/labels");
    Mtx * Theta1 = v.LoadMatrix("data/Theta1");
    Mtx * Theta2 = v.LoadMatrix("data/Theta2");
    Mtx * Theta1_Random = v.LoadMatrix("data/Theta1_Random");
    Mtx * Theta2_Random = v.LoadMatrix("data/Theta2_Random");

    DigitClassifier c;
    c.Train(X, y, Theta1, Theta2, Theta1_Random, Theta2_Random);

  //  std::cout << Theta1->numRows() << std::endl;
  //  std::cout << Theta1->numCols() << std::endl;
   // std::cout << *Theta1 << std::endl;
}
